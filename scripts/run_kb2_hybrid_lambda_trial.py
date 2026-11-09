"""Run one matched 15k hybrid trial without changing historical trainers or banks."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import traceback

from kb2_preference_optimization.artifacts import REPO, atomic_json, sha256
from kb2_preference_optimization.hybrid.bank import Bank
from kb2_preference_optimization.hybrid.source import source_provenance, assert_provenance


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    baseline = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261105_180813_658387'
    native = REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_bg_compare_s42_20261106_051730_406262/native_bg050'
    provenance = source_provenance('yolov8n')
    protected = {}
    for folder, arm in ((baseline, 'dpo'), (native, 'native')):
        done = read(folder/'completed.json')
        assert_provenance(read(folder/'source.json'), provenance)
        assert not done['partial_bank'] and not done['partial_evaluation']
        assert done['arms'][arm]['steps'] == 15000
        for which in ('best', 'last'):
            path = folder/arm/f'{which}.pt'
            digest = sha256(path)
            assert digest == done['arms'][arm][f'{which}_sha256']
            protected[str(path)] = digest
    assert (baseline/'dpo/stream.jsonl').read_bytes() == (native/'native/stream.jsonl').read_bytes()
    bank = Bank(baseline/'preferences')
    assert not bank.manifest['partial_bank']
    assert_provenance(bank.manifest['provenance'], provenance)
    for path in (Path(provenance['source_checkpoint']), Path(provenance['shared_S0']['checkpoint']),
                 bank.root/'manifest.json', bank.root/'pairs.jsonl', bank.root/'contexts.jsonl'):
        protected[str(path)] = sha256(path)
    command = ['bash', str(REPO/'kb2_preference_optimization/run_hybrid_kb3_kb2.sh'),
               '--device', 'cuda', '--steps', '15000', '--eval-interval', '1000', '--seed', '42',
               '--arms', 'dpo', '--dpo-weight', '0.03', '--beta', '0.1', '--lr', '0.000001',
               '--background-weight', '0.5', '--preferences', str(bank.root),
               '--output-root', str(root/'run')]
    root.mkdir(parents=True, exist_ok=False)
    plan = {'created_UTC': datetime.now(timezone.utc).isoformat(), 'baseline': str(baseline),
            'native_control': str(native), 'source': provenance, 'bank': str(bank.root),
            'steps': 15000, 'seed': 42, 'lambda': .03, 'background_weight': .5,
            'selection': 'validation mAP50-95, min_delta=1e-4; TPE step0 eligible',
            'immutable_files': protected, 'command': command, 'test_evaluation': False,
            'runtime': {key: os.environ.get(key) for key in
                        ('CUDA_LAUNCH_BLOCKING', 'PYTHONFAULTHANDLER', 'YOLO_CONFIG_DIR',
                         'MPLCONFIGDIR', 'NO_ALBUMENTATIONS_UPDATE')}}
    atomic_json(root/'plan.json', plan)
    atomic_json(root/'process.json', {'pid': os.getpid(), 'command': command})
    atomic_json(root/'status.json', {'stage': 'training'})
    print(json.dumps({'event': 'lambda_trial_start', 'output': str(root), 'lambda': .03}), flush=True)
    try:
        with (root/'console.log').open('x', buffering=1) as log:
            result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Training exited with {result.returncode}; see console.log')
        done = read(root/'run/completed.json')
        result = done['arms']['dpo']
        assert result['steps'] == 15000 and not done['partial_bank'] and not done['partial_evaluation']
        assert result['reference_unchanged'] and result['frozen_state_unchanged']
        assert result['stream_sha256'] == read(baseline/'completed.json')['arms']['dpo']['stream_sha256']
        assert (root/'run/dpo/stream.jsonl').read_bytes() == (baseline/'dpo/stream.jsonl').read_bytes()
        assert_provenance(read(root/'run/source.json'), provenance)
        assert all(sha256(path) == digest for path, digest in protected.items())
        atomic_json(root/'completed.json', {'plan': plan, 'result': done, 'matched_stream': True,
                                           'protected_files_unchanged': True})
        atomic_json(root/'status.json', {'stage': 'completed', 'best_step': result['best_step']})
        print(json.dumps({'event': 'lambda_trial_completed', 'best_step': result['best_step'],
                          'best_mAP50_95': result['best_metrics']['mAP50_95']}), flush=True)
    except BaseException as error:
        atomic_json(root/'status.json', {'stage': 'failed', 'error': str(error),
                                         'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
