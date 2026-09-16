import sys
import unittest
from pathlib import Path

KB2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KB2_DIR))

from run_ablation import VARIANTS, build_command, parse_args


class AblationTests(unittest.TestCase):
    def test_variants_isolate_sampling_and_loss(self):
        self.assertEqual(VARIANTS['baseline'], {
            'sampling_strategy': 'shuffle', 'feedback_alpha': 0.0})
        self.assertEqual(VARIANTS['sampling'], {
            'sampling_strategy': 'feedback', 'feedback_alpha': 0.0})
        self.assertEqual(VARIANTS['hybrid'], {
            'sampling_strategy': 'feedback', 'feedback_alpha': 0.10})

    def test_commands_share_protocol_and_have_distinct_outputs(self):
        args = parse_args(['--steps', '7', '--seed', '9'])
        commands = {name: build_command(args, name) for name in VARIANTS}
        for command in commands.values():
            self.assertIn('7', command)
            self.assertIn('9', command)
        outputs = [command[command.index('--output') + 1]
                   for command in commands.values()]
        self.assertEqual(len(outputs), len(set(outputs)))


if __name__ == '__main__': unittest.main(verbosity=2)
