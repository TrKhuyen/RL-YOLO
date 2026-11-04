"""Explain fixed candidate-swap audit rejections without loosening its gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
from unittest.mock import patch

from kb2_preference_optimization.artifacts import append_jsonl, atomic_json, sha256
from kb2_preference_optimization.candidate_rank import code_fingerprints
from kb2_preference_optimization.candidate_rank import pairs as module
from kb2_preference_optimization.conditional.bank import Bank
from kb2_preference_optimization.conditional.pairs import matches
from scripts.audit_kb2_conditional_gradients import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = read(args.audit/'completed.json')
    assert manifest['code_sha256'] == code_fingerprints() and not manifest['partial']
    args.output.mkdir(parents=True, exist_ok=False)
    gt_rows = [json.loads(line) for line in (args.audit/'gt_audit.jsonl').read_text().splitlines()]
    ids = sorted({r['image_id'] for r in gt_rows if r['status'] in ('accepted', 'whole_image_gate_rejected', 'valid_capped')})
    bank = Bank(manifest['source_bank'])
    config = module.RankConfig(**manifest['config'])
    original = module.judge_swap
    reasons = Counter()
    not_matched = Counter()
    cap_saturated = Counter()
    current = {}
    counts = Counter()

    def instrumented(context, target, good, bad, g, config):
        valid, evidence = original(context, target, good, bad, g, config)
        a = module.render_swap(context, good, bad, True, config)
        b = module.render_swap(context, good, bad, False, config)
        ag, afp, aa = matches(a, target, config.confidence, config.match_iou)
        bg, bfp, ba = matches(b, target, config.confidence, config.match_iou)
        flags = {
            'chosen_candidate_not_matched_to_target': not any(gt == g and int(a['slots'][p]) == good for p, gt in aa.items()),
            'rejected_candidate_not_matched_to_target': not any(gt == g and int(b['slots'][p]) == bad for p, gt in ba.items()),
            'lost_GT_at_IoU50': not bg <= ag,
            'increased_FP': afp > bfp,
            'target_IoU_gain_too_small': evidence['target_iou_gain'] < config.min_iou_gain-1e-6,
            'total_IoU_gain_too_small': evidence['total_matched_iou_gain'] < config.min_total_iou_gain-1e-6,
            'lost_GT_at_any_IoU50_to95': any(not set(r['rejected_gt']) <= set(r['chosen_gt'])
                                          for r in evidence['threshold_coverage'])}
        assert valid == (not any(flags.values()))
        counts.update(attempts=1, accepted=int(valid), rejected=int(not valid))
        if not valid:
            reasons.update(k for k, flag in flags.items() if flag)
            if flags['chosen_candidate_not_matched_to_target']:
                not_matched.update({'good_not_present_after_NMS': int(good not in evidence['chosen_slots']),
                                    'good_present_but_not_target_match': int(good in evidence['chosen_slots'])})
                cap_saturated.update(good_missing_with_cap_saturated=int(
                    good not in evidence['chosen_slots'] and len(evidence['chosen_slots']) >= config.max_det))
        append_jsonl(args.output/'attempts.jsonl', {
            'image_id': current['image_id'], 'target_gt': g, 'class_id': int(target['labels'][g]),
            'good_slot': good, 'bad_slot': bad, 'valid': valid,
            'rejection_flags': flags, 'evidence': evidence})
        return valid, evidence

    with patch.object(module, 'judge_swap', instrumented):
        for image_id in ids:
            _, target = bank.dataset[image_id]
            current['image_id'] = image_id
            module.build_rank_context(bank.contexts[image_id], target, config)
    assert counts['attempts'] == manifest['totals']['attempted_pairs']
    assert counts['accepted'] == manifest['totals']['valid_before_cap']
    assert counts['rejected'] == manifest['totals']['gate_rejected']
    accepted = [json.loads(s) for s in (args.audit/'pairs.jsonl').read_text().splitlines()]
    result = {'source_audit': str(args.audit.resolve()), 'config': manifest['config'],
              'counts': dict(counts), 'overlapping_rejection_reasons': dict(reasons),
              'chosen_not_matched_breakdown': dict(not_matched),
              'good_missing_with_cap_saturated': cap_saturated['good_missing_with_cap_saturated'],
              'GT_status_counts': dict(Counter(r['status'] for r in gt_rows)),
              'accepted_class_counts': dict(Counter(str(r['class_id']) for r in accepted)),
              'accepted_train_images': len({r['image_id'] for r in accepted if not r['probe']}),
              'accepted_train_classes': sorted({r['class_id'] for r in accepted if not r['probe']}),
              'scope': 'Replayed identical fixed proposals; rejection reasons overlap; no thresholds changed',
              'script_sha256': sha256(Path(__file__))}
    atomic_json(args.output/'analysis.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
