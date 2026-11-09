"""Schedules count optimizer updates, independently of validation frequency."""
import math


def validate_schedule(lr_drop_step, lr_drop_factor, early_eval_interval, early_eval_until):
    if (min(lr_drop_step, early_eval_interval, early_eval_until) < 0
            or not math.isfinite(lr_drop_factor) or not 0 < lr_drop_factor <= 1
            or (early_eval_interval > 0 and early_eval_until < 1)):
        raise ValueError('Invalid learning-rate or early-evaluation schedule')


def learning_rate_for_update(base_lr, update, lr_drop_step=0, lr_drop_factor=.1):
    if not math.isfinite(base_lr) or base_lr <= 0 or update < 1:
        raise ValueError('A positive learning rate and update index are required')
    validate_schedule(lr_drop_step, lr_drop_factor, 0, 0)
    return base_lr * (lr_drop_factor if lr_drop_step > 0 and update > lr_drop_step else 1.)


def evaluation_steps(steps, eval_interval, early_eval_interval=0, early_eval_until=2000):
    if min(steps, eval_interval) < 1:
        raise ValueError('Positive budgets required')
    validate_schedule(0, .1, early_eval_interval, early_eval_until)
    grid = {0, steps, *range(eval_interval, steps + 1, eval_interval)}
    if early_eval_interval:
        grid.update(range(early_eval_interval, min(steps, early_eval_until) + 1, early_eval_interval))
    return sorted(grid)
