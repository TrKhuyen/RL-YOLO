"""One scheduled DPO pipeline with fresh S0/TPE comparisons."""
import json
from pathlib import Path

import torch

from ..artifacts import atomic_json, source_checkpoint as supervised_checkpoint
from ..conditional.evaluation import evaluate
from ..conditional.policy import ConditionalAdapter, configure_numerics
from ..data import detection_loader
from ..hybrid import EXPERIMENT_VERSION
from ..hybrid.bank import Bank, generate
from ..hybrid.run import deltas
from ..hybrid.source import assert_provenance, source_provenance
from . import TRAINING_VARIANT
from .train import train


def execute(args, output, provenance):
    if args.arms != ['dpo']:
        raise ValueError('The scheduled workflow supports one DPO pipeline')
    atomic_json(output/'request.json', {k: str(v) if isinstance(v, Path) else v
                                       for k, v in vars(args).items()})
    atomic_json(output/'source.json', provenance)
    print(json.dumps({'event': 'start', 'experiment_version': EXPERIMENT_VERSION,
                      'training_variant': TRAINING_VARIANT, 'output_root': str(output),
                      'arms': ['dpo'], 'steps': args.steps,
                      'lr_drop_after_update': args.lr_drop_step,
                      'early_eval_interval': args.early_eval_interval}), flush=True)
    bank_path = args.preferences.resolve() if args.preferences else output/'preferences'
    if not args.preferences:
        manifest = generate(bank_path, args.device, args.limit_images)
        print(json.dumps({'event': 'bank_complete', 'audit': manifest['audit']['totals']}), flush=True)
    bank = Bank(bank_path)
    assert_provenance(provenance, bank.manifest['provenance'])
    atomic_json(output/'bank_audit.json', bank.manifest['audit'])
    del bank
    if args.audit_only:
        return
    configure_numerics()
    baseline = ConditionalAdapter(supervised_checkpoint('yolov8n'), args.device)
    baseline.model.requires_grad_(False)
    s0_metrics, _ = evaluate(baseline, detection_loader('valid', 640, 8), image_limit=args.eval_limit)
    atomic_json(output/'S0_validation.json', s0_metrics)
    del baseline
    if torch.device(args.device).type == 'cuda':
        torch.cuda.empty_cache()
    arm_dir = output/'dpo'
    result = train(bank_path, arm_dir, device=args.device, steps=args.steps, seed=args.seed,
                   eval_interval=args.eval_interval, eval_limit=args.eval_limit,
                   beta=args.beta, dpo_weight=args.dpo_weight, lr=args.lr,
                   audit_pairs=args.audit_pairs, background_weight=args.background_weight,
                   lr_drop_step=args.lr_drop_step, lr_drop_factor=args.lr_drop_factor,
                   early_eval_interval=args.early_eval_interval, early_eval_until=args.early_eval_until)
    source = json.loads((arm_dir/'source_validation.json').read_text())
    rows = [json.loads(line) for line in (arm_dir/'metrics.jsonl').read_text().splitlines()]
    best = next(r['metrics'] for r in rows if r['event'] == 'validation' and r['step'] == result['best_step'])
    result.update(best_checkpoint=str(arm_dir/'best.pt'), last_checkpoint=str(arm_dir/'last.pt'),
                  best_metrics=best, best_delta_vs_TPE=deltas(best, source),
                  best_delta_vs_S0=deltas(best, s0_metrics),
                  last_delta_vs_TPE=deltas(result['last_metrics'], source),
                  last_delta_vs_S0=deltas(result['last_metrics'], s0_metrics))
    assert_provenance(provenance, source_provenance('yolov8n'))
    comparison = {'experiment_version': EXPERIMENT_VERSION, 'training_variant': TRAINING_VARIANT,
                  'split': 'valid', 'output_root': str(output), 'source': provenance['source_tpe'],
                  'matched_stream': True, 'partial_bank': result['partial_bank'],
                  'partial_evaluation': bool(args.eval_limit), 'S0': s0_metrics, 'TPE': source,
                  'arms': {'dpo': result},
                  'delta_units': 'AP/P/R/F1 fractions; multiply by 100 for percentage points; FP/FN counts'}
    atomic_json(output/'comparison.json', comparison)
    atomic_json(output/'completed.json', comparison)
    print(json.dumps({'event': 'completed', 'output_root': str(output),
                      'best_step': result['best_step'], 'best_mAP50_95': best['mAP50_95']}), flush=True)
