"""Matched 15k TPE continuations: native/bg=.5 and native+DPO/bg=.25."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

from kb2_preference_optimization.artifacts import REPO, atomic_json, sha256
from kb2_preference_optimization.hybrid.source import source_provenance, assert_provenance
from kb2_preference_optimization.hybrid.bank import Bank


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--continue-completed', action='store_true',
                        help='Reuse verified completed arms; archive incomplete arms and restart them from TPE')
    args = parser.parse_args()
    output = args.output.resolve()
    baseline = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261105_180813_658387'
    old = read(baseline/'comparison.json')
    provenance = source_provenance('yolov8n')
    assert_provenance(read(baseline/'source.json'), provenance)
    bank = Bank(baseline/'preferences')
    if bank.manifest['partial_bank'] or old['partial_evaluation']:
        raise ValueError('A complete bank and full baseline validation are required')
    protected = {str(baseline/'dpo'/f'{name}.pt'): sha256(baseline/'dpo'/f'{name}.pt')
                 for name in ('best', 'last')}
    protected.update({provenance['source_checkpoint']: provenance['source_sha256'],
                      provenance['shared_S0']['checkpoint']: provenance['shared_S0']['sha256'],
                      str(bank.root/'manifest.json'): sha256(bank.root/'manifest.json'),
                      str(bank.root/'pairs.jsonl'): bank.manifest['pairs_sha256'],
                      str(bank.root/'contexts.jsonl'): bank.manifest['contexts_sha256']})
    if output.exists() and not args.continue_completed:
        raise FileExistsError('Use a fresh campaign or explicitly continue completed arms')
    output.mkdir(parents=True, exist_ok=args.continue_completed)
    plan = {'created_UTC': datetime.now(timezone.utc).isoformat(), 'baseline': str(baseline),
            'seed': 42, 'steps_per_arm': 15000, 'eval_interval': 1000,
            'source': provenance, 'bank': str(bank.root), 'immutable_files': protected,
            'arms': [{'name': 'native_bg050', 'arm': 'native', 'background_weight': .5,
                      'effective_DPO_lambda': 0.},
                     {'name': 'dpo_bg025', 'arm': 'dpo', 'background_weight': .25,
                      'effective_DPO_lambda': .1}],
            'selection': 'validation mAP50-95, min_delta=1e-4; TPE step0 eligible',
            'test_evaluation': False}
    if args.continue_completed:
        saved_plan = read(output/'plan.json')
        assert_provenance(saved_plan['source'], provenance)
        if saved_plan['arms'] != plan['arms'] or saved_plan['bank'] != plan['bank']:
            raise ValueError('Continuation plan differs')
        if not all(sha256(p) == h for p, h in saved_plan['immutable_files'].items()):
            raise ValueError('Protected files changed since the initial plan')
        plan = saved_plan
        protected = plan['immutable_files']
    else:
        atomic_json(output/'plan.json', plan)
    atomic_json(output/'status.json', {'stage': 'starting', 'completed_arms': []})
    completed = {}
    try:
        for spec in plan['arms']:
            name = spec['name']
            existing = output/name/'completed.json'
            if args.continue_completed and existing.exists():
                cached = read(existing)
                result = cached['arms'][spec['arm']]
                manifest = read(output/name/spec['arm']/'manifest.json')
                assert_provenance(manifest['provenance'], provenance)
                if (result['steps'] != 15000 or cached['partial_bank'] or cached['partial_evaluation']
                        or result['stream_sha256'] != old['arms']['dpo']['stream_sha256']
                        or manifest['config']['native_background_weight'] != spec['background_weight']
                        or manifest['config']['effective_dpo_weight'] != spec['effective_DPO_lambda']
                        or any(sha256(output/name/spec['arm']/f'{which}.pt') != result[f'{which}_sha256']
                               for which in ('best', 'last'))):
                    raise ValueError(f'Completed arm cannot be reused: {name}')
                completed[name] = cached
                print(json.dumps({'event': 'arm_reused_completed', 'name': name}), flush=True)
                continue
            if args.continue_completed and (output/name).exists():
                stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
                archive = output/f'{name}_interrupted_{stamp}'
                (output/name).rename(archive)
                console = output/f'{name}.console.log'
                if console.exists():
                    console.rename(output/f'{name}_interrupted_{stamp}.console.log')
                with (output/'restart_events.jsonl').open('a') as log:
                    log.write(json.dumps({'arm': name, 'archived': str(archive),
                                          'restart': 'fresh TPE initialization, same seed/recipe'})+'\n')
            command = ['bash', str(REPO/'kb2_preference_optimization/run_hybrid_kb3_kb2.sh'),
                       '--device', args.device, '--steps', '15000', '--eval-interval', '1000',
                       '--seed', '42', '--arms', spec['arm'], '--dpo-weight', '0.1', '--beta', '0.1',
                       '--lr', '0.000001', '--background-weight', str(spec['background_weight']),
                       '--preferences', str(bank.root), '--output-root', str(output/name)]
            atomic_json(output/'status.json', {'stage': 'training', 'active_arm': name,
                                              'completed_arms': list(completed), 'command': command})
            print(json.dumps({'event': 'arm_start', 'name': name, 'output': str(output/name)}), flush=True)
            with (output/f'{name}.console.log').open('x', buffering=1) as log:
                result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'{name} failed with exit {result.returncode}; see {name}.console.log')
            done = read(output/name/'completed.json')
            arm_result = done['arms'][spec['arm']]
            if (arm_result['steps'] != 15000 or done['partial_bank'] or done['partial_evaluation']
                    or arm_result['stream_sha256'] != old['arms']['dpo']['stream_sha256']):
                raise ValueError(f'Budget, data coverage or stream differs for {name}')
            assert_provenance(read(output/name/'source.json'), provenance)
            completed[name] = done
            print(json.dumps({'event': 'arm_complete', 'name': name, 'best_step': arm_result['best_step'],
                              'best_mAP50_95': arm_result['best_metrics']['mAP50_95']}), flush=True)
        if not all(sha256(p) == h for p, h in protected.items()):
            raise ValueError('A protected source/baseline/bank file changed')
        atomic_json(output/'completed.json', {'plan': plan, 'matched_stream': True,
                                            'protected_files_unchanged': True, 'runs': completed})
        atomic_json(output/'status.json', {'stage': 'completed', 'completed_arms': list(completed)})
        print(json.dumps({'event': 'comparison_runs_completed', 'output': str(output)}), flush=True)
    except BaseException as error:
        atomic_json(output/'status.json', {'stage': 'failed', 'completed_arms': list(completed),
                                          'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
