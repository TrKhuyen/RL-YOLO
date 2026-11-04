"""Run generation plus three matched 2k-step recovery pilots on CUDA."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kb2_preference_optimization.artifacts import atomic_json


def command(root, name, arguments):
    with (root / f'{name}.log').open('w') as stream:
        child = subprocess.Popen([sys.executable, '-m', *arguments], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        for line in child.stdout:
            stream.write(line)
            stream.flush()
            print(line.rstrip(), flush=True)
        result = child.wait()
    if result:
        raise RuntimeError(f'{name} failed; see {root / (name + ".log")}')


def main(root):
    root = root.resolve()
    if not (root / 'audit.json').exists():
        if root.exists():
            raise FileExistsError('Incomplete generation; use a fresh directory')
        subprocess.run([sys.executable, '-m', 'kb2_preference_optimization.recovery.generate',
                        '--output', str(root), '--device', 'cuda'], cwd=ROOT, check=True)
    for arm in ('control', 'selector', 'balanced'):
        folder = root / f'pilot_{arm}'
        if not (folder / 'completed.json').exists():
            command(root, arm, ['kb2_preference_optimization.recovery.train_pilot',
                               '--root', str(root), '--arm', arm, '--steps', '2000', '--device', 'cuda'])
    completions = {arm: json.loads((root / f'pilot_{arm}/completed.json').read_text())
                   for arm in ('control', 'selector', 'balanced')}
    assert len({r['native_sequence_sha256'] for r in completions.values()}) == 1
    assert all(r['reference_immutable'] for r in completions.values())
    atomic_json(root / 'pilot_completed.json', {
        'status': 'complete', 'steps': 2000, 'seed': 42, 'pilot': True,
        'arms': ['control', 'selector', 'balanced'], 'native_sequence_identical': True,
        'all_reference_immutable': True, 'test_evaluated': False})
    print('RECOVERY_PILOT_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root)
