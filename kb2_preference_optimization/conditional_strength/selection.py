"""Multimetric review without counting dependent FP/precision twice."""
KEYS = ('mAP50_95', 'precision', 'operating_recall')


def dominates(first, second):
    return all(first[k] >= second[k] for k in KEYS) and any(first[k] > second[k] for k in KEYS)


def select_pareto(history):
    return [r['step'] for r in history
            if not any(dominates(other['metrics'], r['metrics']) for other in history)]
