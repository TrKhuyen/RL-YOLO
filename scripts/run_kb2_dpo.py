"""Current KB2 entrypoint: the complete DPO pipeline, without control arms.

Historical engines stay byte-identical so saved banks/checkpoints remain replayable.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization', choices=['tpe', 's0'], default='tpe')
    parser.add_argument('--model', choices=['yolov8n'], default='yolov8n')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-interval', type=int, default=1000)
    parser.add_argument('--arms', nargs='+', choices=['dpo'], default=['dpo'],
                        help='Compatibility with earlier commands; only one DPO pipeline is supported')
    parser.add_argument('--dpo-weight', type=float)
    parser.add_argument('--beta', type=float, default=.1)
    parser.add_argument('--background-weight', type=float)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--lr-drop-step', type=int, default=0,
                        help='Keep the initial LR through this update; 0 disables the drop')
    parser.add_argument('--lr-drop-factor', type=float, default=.1)
    parser.add_argument('--early-eval-interval', type=int, default=0)
    parser.add_argument('--early-eval-until', type=int, default=2000)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--preferences', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--limit-images', type=int, default=0)
    parser.add_argument('--eval-limit', type=int, default=0)
    parser.add_argument('--audit-pairs', type=int, default=32)
    parser.add_argument('--fp-tolerance', type=float)
    parser.add_argument('--ap-tolerance', type=float)
    args = parser.parse_args(argv)
    args.steps = args.steps if args.steps is not None else (15000 if args.initialization == 'tpe' else 2000)
    args.dpo_weight = args.dpo_weight if args.dpo_weight is not None else (.1 if args.initialization == 'tpe' else .01)
    if args.initialization == 's0' and (args.lr_drop_step or args.early_eval_interval):
        parser.error('Scheduled updates currently support TPE initialization only')
    if args.initialization == 's0' and args.background_weight is not None:
        parser.error('The historical S0 recipe does not accept --background-weight')
    if args.initialization == 'tpe' and (args.fp_tolerance is not None or args.ap_tolerance is not None):
        parser.error('TPE continuation selects by mAP50-95; recall gate options apply only to the historical S0 recipe')
    args.background_weight = .5 if args.background_weight is None else args.background_weight
    args.fp_tolerance = .05 if args.fp_tolerance is None else args.fp_tolerance
    args.ap_tolerance = .001 if args.ap_tolerance is None else args.ap_tolerance
    if (args.arms != ['dpo'] or min(args.steps, args.eval_interval, args.audit_pairs) < 1
            or min(args.seed, args.limit_images, args.eval_limit) < 0
            or not all(math.isfinite(v) and v > 0 for v in (args.beta, args.dpo_weight, args.lr))
            or min(args.lr_drop_step, args.early_eval_interval, args.early_eval_until) < 0
            or not math.isfinite(args.lr_drop_factor) or not 0 < args.lr_drop_factor <= 1
            or (args.early_eval_interval > 0 and args.early_eval_until < 1)
            or not math.isfinite(args.background_weight) or not 0 <= args.background_weight <= 1
            or not all(math.isfinite(v) and v >= 0 for v in (args.fp_tolerance, args.ap_tolerance))
            or (args.preferences and args.limit_images)):
        parser.error('Invalid DPO budget/weights; exactly one DPO pipeline is required')
    return args


def run_hybrid(args):
    import torch
    from kb2_preference_optimization.artifacts import atomic_json, sha256
    from kb2_preference_optimization.hybrid.run import execute, Tee
    from kb2_preference_optimization.hybrid.source import source_provenance
    from kb2_preference_optimization.hybrid import EXPERIMENT_VERSION

    scheduled = bool(args.lr_drop_step or args.early_eval_interval)
    if scheduled:
        from kb2_preference_optimization.hybrid_schedule.run import execute

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    provenance = source_provenance('yolov8n')
    if args.check_only:
        print(json.dumps({'event': 'preflight_pass', 'workflow': 'kb2_dpo_pipeline',
                          'experiment_version': EXPERIMENT_VERSION, 'device': str(device),
                          'arms': ['dpo'], 'source': provenance['source_tpe'],
                          'scheduled_updates': scheduled,
                          'shared_S0': provenance['shared_S0'],
                          'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}))
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    trial = provenance['source_tpe']['trial']
    output = (args.output_root or Path('kb2_preference_optimization/outputs') /
              f'yolov8n_hybrid_tpe{trial}_s{args.seed}_{stamp}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'workflow.json', {'workflow': 'kb2_dpo_pipeline',
                                       'entrypoint_sha256': sha256(Path(__file__)),
                                       'evaluation_models': ['S0', 'TPE', 'KB2_DPO'],
                                       'scheduled_updates': scheduled,
                                       'control_arms': [], 'native_detection_loss_retained': True})
    with (output/'run.log').open('x', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            execute(args, output, provenance)
            if not args.audit_only:
                comparison = json.loads((output/'comparison.json').read_text())
                comparison.update(workflow='kb2_dpo_pipeline',
                                  evaluation_models=['S0', 'TPE', 'KB2_DPO'],
                                  interpretation='KB2 DPO denotes the complete preference-generation and '
                                  'native-detection-plus-DPO training pipeline. Compare the final pipeline '
                                  'with S0 and TPE; these comparisons do not isolate individual loss terms.')
                atomic_json(output/'comparison.json', comparison)
                atomic_json(output/'completed.json', comparison)


def run_s0(args):
    from kb2_preference_optimization.conditional.run import main as historical_main
    forwarded = ['--model', args.model, '--device', args.device, '--arms', 'dpo']
    for name in ('steps', 'seed', 'eval_interval', 'eval_limit', 'limit_images', 'audit_pairs',
                 'beta', 'dpo_weight', 'lr', 'fp_tolerance', 'ap_tolerance', 'output_root', 'preferences'):
        value = getattr(args, name)
        if value is not None:
            forwarded.extend(['--'+name.replace('_', '-'), str(value)])
    for name in ('check_only', 'audit_only'):
        if getattr(args, name):
            forwarded.append('--'+name.replace('_', '-'))
    original = sys.argv
    try:
        sys.argv = [original[0], *forwarded]
        historical_main()
    finally:
        sys.argv = original


def main(argv=None):
    args = parse_args(argv)
    if args.initialization == 'tpe':
        run_hybrid(args)
    else:
        run_s0(args)


if __name__ == '__main__':
    main()
