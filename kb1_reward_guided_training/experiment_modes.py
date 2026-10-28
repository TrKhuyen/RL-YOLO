"""Effective objective coefficients shared by the runner and experiment planning."""
MODES = ('native_only', 'proxy_only', 'reward_only', 'kb1b', 'kb1b_no_sp')


def objective_weights(mode, config):
    if mode not in MODES:
        raise ValueError(f'Unknown objective mode: {mode}')
    return {
        'native': config['native_supervised_loss_weight'],
        'proxy': config['supervised_loss_weight'] if mode in ('proxy_only', 'kb1b', 'kb1b_no_sp') else 0.,
        'reward': config['reward_loss_weight'] if mode in ('reward_only', 'kb1b', 'kb1b_no_sp') else 0.,
        'stability': 0. if mode == 'kb1b_no_sp' else config['stability_loss_weight'],
    }
