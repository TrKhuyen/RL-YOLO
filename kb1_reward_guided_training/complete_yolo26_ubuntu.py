"""Finish seed-42 YOLO26 with fresh Ubuntu runs or validated historical guided."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT.parent / '.kb1-tools/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / '.kb1-tools/matplotlib'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh directory for preflight, native training, and evaluation')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fresh-pair', action='store_true', help='Train both native-only and guided on Ubuntu')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from run_provenance import runtime_manifest, dataset_manifest, sha256
    from evaluation_provenance import inspect_checkpoint
    from evaluate import canonical_evaluate_model
    from dataloader import get_pest_dataloader
    from train_rl import CHECKPOINTS
    from run_verified import atomic_json
    import torch
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('YOLO26 completion requires a working CUDA GPU')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    historic = ROOT / 'checkpoint_reward_guide_trainning/screening_seed42_v1/yolo26n/kb1b'
    source = CHECKPOINTS['yolo26n']
    record, _ = inspect_checkpoint('yolo26n', historic / 'best.pt', source, 42, 'kb1b')
    runtime = runtime_manifest()
    # The archived Windows package is additional evidence, never imported on Linux.
    archived = ROOT.parent / '.venv/Lib/site-packages/ultralytics'
    for relative, digest in ([] if args.fresh_pair else runtime['ultralytics_code_sha256'].items()):
        if not (archived / relative).is_file() or sha256(archived / relative) != digest:
            raise RuntimeError(f'Historical Ultralytics implementation unavailable/different: {relative}; rerun a fresh S1/G pair')
    loader = get_pest_dataloader(dataset_manifest()['root'], 'val', batch_size=16,
                                 num_workers=0, shuffle=False)
    completed = json.loads((historic / 'completed.json').read_text())
    parity = {}
    parity_matches = True
    for stage, checkpoint, base, expected in [
        ('supervised', source, None, completed['baseline']),
        ('guided', historic / 'best.pt', source, completed['best']),
    ]:
        print('Checking historical validation parity:', stage, flush=True)
        actual = canonical_evaluate_model('yolo26n', str(checkpoint), loader,
                                          device=args.device, supervised_ckpt=base)
        parity[stage] = actual
        for metric in ('mAP50', 'mAP50_95', 'AR300'):
            if abs(actual[metric] - expected[metric]) > 1e-6:
                parity_matches = False
                print(f'Historical metric differs: {stage}/{metric}: {actual[metric]} vs {expected[metric]}', flush=True)
    atomic_json(out / 'validation_parity.json', parity)
    if not parity_matches and not args.fresh_pair:
        raise RuntimeError('Historical parity failed; rerun with --fresh-pair in a new output')
    atomic_json(out / 'manifest.json', {'runtime': runtime, 'dataset': dataset_manifest(),
        'historical_guided': record, 'fresh_pair': args.fresh_pair, 'historical_parity_matches': parity_matches,
        'note': 'Both S1/G are retrained on Ubuntu.' if args.fresh_pair else 'Native-only runs on Ubuntu; guided is a frozen Windows seed-42 artifact after validation parity checks.'})
    del loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    for mode in ('native_only', 'kb1b'):
        command = [sys.executable, '-u', str(ROOT / 'run_verified.py'), '--model', 'yolo26n',
                   '--mode', mode, '--output', str(out / 'preflight'),
                   '--steps', '2', '--eval-interval', '2', '--workers', '0']
        print('Preflight:', mode, flush=True)
        subprocess.run(command, check=True, cwd=ROOT.parent)
    for mode in (('native_only', 'kb1b') if args.fresh_pair else ('native_only',)):
        print('Training YOLO26n seed 42:', mode, flush=True)
        subprocess.run([sys.executable, '-u', str(ROOT / 'run_verified.py'), '--model', 'yolo26n',
                        '--mode', mode, '--seed', '42', '--workers', '0',
                        '--output', str(out / 'screening')], check=True, cwd=ROOT.parent)
    if not args.fresh_pair:
        # Expose the frozen historical run without copying or rewriting it.
        os.symlink(historic, out / 'screening/yolo26n/kb1b', target_is_directory=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'evaluate.py'), '--model', 'YOLO26n',
                    '--split', 'test', '--seed', '42', '--workers', '0',
                    '--screening-root', str(out / 'screening'),
                    '--output', str(out / 'test')], check=True, cwd=ROOT.parent)
    atomic_json(out / 'completed.json', {'status': 'complete', 'seed': 42,
        'test': str(out / 'test'), 'historical_guided_reused': not args.fresh_pair})
    print('YOLO26 completion successful:', out, flush=True)


if __name__ == '__main__':
    main()
