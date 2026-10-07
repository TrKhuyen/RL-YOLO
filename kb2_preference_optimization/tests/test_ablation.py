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
        self.assertEqual(VARIANTS['pairwise'], {
            'sampling_strategy': 'shuffle', 'feedback_alpha': 0.0})

    def test_pairwise_requires_positive_weight_but_baseline_allows_zero(self):
        self.assertEqual(parse_args(['--variants', 'baseline',
                                     '--pairwise-alpha', '0']).pairwise_alpha, 0)
        with self.assertRaises(SystemExit):
            parse_args(['--variants', 'pairwise', '--pairwise-alpha', '0'])

    def test_relative_paths_become_absolute_before_child_changes_cwd(self):
        import os
        original = Path.cwd()
        try:
            os.chdir(KB2_DIR)
            args = parse_args([
                '--checkpoint', '../kb1_reward_guided_training/checkpoint_based/yolov8n/weights/best.pt',
                '--data-root', '../pre-data/data/v2i_cleanned',
                '--feedback', 'feedback_data_clean/sample.jsonl',
            ])
            command = build_command(args, 'hybrid')
            for option in ('--checkpoint', '--data-root', '--feedback'):
                self.assertTrue(Path(command[command.index(option) + 1]).is_absolute())
        finally:
            os.chdir(original)

    def test_commands_share_protocol_and_have_distinct_outputs(self):
        args = parse_args(['--steps', '7', '--seed', '9'])
        commands = {name: build_command(args, name) for name in VARIANTS}
        for command in commands.values():
            self.assertIn('7', command)
            self.assertIn('9', command)
            self.assertEqual(command[command.index('--val-conf') + 1], '0.001')
            self.assertEqual(command[command.index('--img-size') + 1], '640')
            self.assertTrue(command[command.index('--checkpoint') + 1].endswith('best.pt'))
        self.assertEqual(
            commands['pairwise'][commands['pairwise'].index('--pairwise-alpha') + 1],
            '0.01')
        for name in ('baseline', 'sampling', 'hybrid'):
            self.assertEqual(
                commands[name][commands[name].index('--pairwise-alpha') + 1],
                '0.0')
        outputs = [command[command.index('--output') + 1]
                   for command in commands.values()]
        self.assertEqual(len(outputs), len(set(outputs)))


if __name__ == '__main__': unittest.main(verbosity=2)
