"""Check whether proposal caps explain low swap coverage, keeping every quality gate."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from kb2_preference_optimization.artifacts import atomic_json, sha256
from kb2_preference_optimization.candidate_rank import code_fingerprints
from kb2_preference_optimization.candidate_rank.pairs import RankConfig, build_rank_context
from kb2_preference_optimization.conditional.bank import Bank, probe_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.audit/'completed.json').read_text())
    assert not manifest['partial'] and code_fingerprints() == manifest['code_sha256']
    gt_rows = [json.loads(s) for s in (args.audit/'gt_audit.jsonl').read_text().splitlines()]
    ids = sorted({r['image_id'] for r in gt_rows if r['status'] in ('accepted', 'whole_image_gate_rejected', 'valid_capped')})
    bank = Bank(manifest['source_bank'])
    config = replace(RankConfig(**manifest['config']), attempts_per_gt=1000000, max_pairs_per_image=1000000)
    counts, records = Counter(), []
    for image_id in ids:
        _, target = bank.dataset[image_id]
        _, pairs, diagnostics, _ = build_rank_context(bank.contexts[image_id], target, config)
        counts.update(attempts=diagnostics['attempted_pairs'], rejected=diagnostics['gate_rejected'],
                      valid=diagnostics['valid_before_cap'])
        for r in pairs:
            r.update(image_id=image_id, image_path=target['image_path'], probe=probe_image(target['image_path']))
            records.append(r)
            counts.update(train_pairs=int(not r['probe']), probe_pairs=int(r['probe']))
    result = {'baseline_pairs': manifest['totals']['pairs'],
              'baseline_train_pairs': manifest['totals']['train_pairs'],
              'all_alternatives': dict(counts), 'images_replayed': len(ids),
              'pair_images': len({r['image_id'] for r in records}),
              'train_images': len({r['image_id'] for r in records if not r['probe']}),
              'train_classes': sorted({r['class_id'] for r in records if not r['probe']}),
              'config': config.manifest(), 'script_sha256': sha256(Path(__file__)),
              'scope': 'Proposal-cap sensitivity only; all quality gates unchanged, no training'}
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'proposal_cap_sensitivity.json', result)
    (args.output/'all_alternatives_pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
