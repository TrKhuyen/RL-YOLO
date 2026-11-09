"""Audit saved KB1 gains and optionally compare precision at matched validation recall.

No training, test inference, or historical artifact modifications are performed.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / 'kb1_reward_guided_training'
RUN = KB / 'checkpoint_reward_guide_trainning/remaining_seed43_44_ablation_v1'
sys.path.insert(0, str(KB))
FIELDS = ('mAP50_95', 'mAP50', 'precision', 'operating_recall', 'f1',
          'macro_f1', 'AR300', 'APs', 'APm', 'APl')


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty table: {path}')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def saved_results(output):
    with (RUN / 'results_all.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    groups = defaultdict(dict)
    for row in rows:
        key = row['experiment'], row['split'], row['Model'], int(row['run_seed'])
        if row['Stage'] in groups[key]:
            raise ValueError(f'Duplicate result: {key}/{row["Stage"]}')
        groups[key][row['Stage']] = row
    observations, summaries = [], defaultdict(list)
    for (experiment, split, model, seed), stages in groups.items():
        for stage, row in stages.items():
            total = 0
            for name in row['classes'].strip('[]').split(','):
                if name.strip():
                    total += 1
            if total != 28:
                raise ValueError('Unexpected class set')
            for metric in FIELDS:
                value = float(row[metric]) * 100
                base = float(stages['supervised'][metric]) * 100
                summaries[experiment, split, model, stage, metric].append((value, base))
            # P/R are micro counts on the same GT set, so counts can be recovered.
            n_gt = 1063 if split == 'test' else 1887
            tp_float = float(row['operating_recall']) * n_gt
            fp_float = tp_float / float(row['precision']) - tp_float
            if max(abs(tp_float-round(tp_float)), abs(fp_float-round(fp_float))) > 1e-6:
                raise ValueError('P/R cannot be reconstructed as integer counts')
            observations.append({'experiment': experiment, 'split': split, 'Model': model,
                                 'seed': seed, 'stage': stage, 'TP': round(tp_float),
                                 'FP': round(fp_float), 'FN': n_gt-round(tp_float)})
    result = []
    for (experiment, split, model, stage, metric), values in sorted(summaries.items()):
        changes = [v-b for v, b in values]
        result.append({'experiment': experiment, 'split': split, 'Model': model,
                       'stage': stage, 'metric': metric, 'n': len(values),
                       'mean_pct': statistics.mean(v for v, _ in values),
                       'S0_pct': statistics.mean(b for _, b in values),
                       'delta_pp': statistics.mean(changes),
                       'delta_sample_sd_pp': statistics.stdev(changes)})
    write_csv(output / 'saved_metrics.csv', result)
    write_csv(output / 'reconstructed_counts.csv', observations)
    return rows


def ranked_points(preds, targets):
    """Exact score prefixes with same-class one-to-one IoU=.5 matching.

    Adding lower-score predictions cannot change earlier greedy assignments.
    Equal scores are grouped so every point corresponds to an attainable threshold.
    """
    import torch
    from torchvision.ops import box_iou
    bins = defaultdict(lambda: [0, 0])
    n_gt = sum(len(t['labels']) for t in targets)
    for pred, target in zip(preds, targets):
        gt_boxes, gt_labels = target['boxes'], target['labels']
        scores, labels = pred['scores'], pred['labels']
        ious = box_iou(gt_boxes.float(), pred['boxes'].float())
        used = set()
        for idx in scores.argsort(descending=True).tolist():
            candidates = [j for j in range(len(gt_boxes))
                          if j not in used and int(gt_labels[j]) == int(labels[idx])]
            matched = False
            if candidates:
                pos = int(ious[candidates, idx].argmax())
                gt = candidates[pos]
                if float(ious[gt, idx]) >= .5:
                    used.add(gt)
                    matched = True
            bins[float(scores[idx])][0 if matched else 1] += 1
    tp, fp = 0, 0
    points = []
    for threshold, (nt, nf) in sorted(bins.items(), reverse=True):
        tp += nt
        fp += nf
        points.append({'threshold': threshold, 'TP': tp, 'FP': fp, 'FN': n_gt-tp,
                       'precision': tp/max(tp+fp, 1), 'recall': tp/max(n_gt, 1),
                       'f1': 2*tp/max(2*tp+fp+n_gt-tp, 1)})
    if not points:
        raise ValueError('No validation predictions')
    return points


def validation(output, seeds, device, reuse=None, allow_backend_difference=False):
    import torch
    from canonical_eval import _operating_point_counts, _summarize_counts
    from dataloader import get_pest_dataloader
    from evaluation_provenance import inspect_checkpoint
    from run_provenance import DATA_ROOT, dataset_manifest, runtime_manifest
    from train_rl import CHECKPOINTS, load_adapter
    from torchmetrics.detection import MeanAveragePrecision

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = dataset_manifest()
    expected = json.loads((RUN / 'plan.json').read_text())['dataset']
    if dataset['sha256'] != expected['sha256']:
        raise ValueError('Dataset changed since KB1 experiment')
    source = CHECKPOINTS['yolov8s']
    jobs = [(0, 'supervised', source)]
    jobs.extend((seed, stage, RUN / f'ablation/seed{seed}/yolov8s/{stage}/last.pt')
                for seed in seeds for stage in ('native_only', 'proxy_only', 'kb1b'))
    manifest = {'dataset': dataset, 'runtime': runtime_manifest(), 'device': device,
                'split': 'val', 'model': 'yolov8s', 'selection': 'ablation_last_10000',
                'protocol': {'input': 'FP32 640 letterbox', 'conf': .001, 'nms_iou': .60,
                             'max_det': 300, 'operating_conf': .25, 'operating_iou': .5},
                'AP_parity_tolerance_fraction': 1e-4,
                'historical_parity_enforced': not allow_backend_difference,
                'comparison_scope': 'All checkpoints evaluated in one fresh backend cohort; do not pool with historical CUDA.',
                'inputs': []}
    evaluations, all_points = [], []
    baseline_tp = None
    for seed, stage, checkpoint in jobs:
        record, payload = inspect_checkpoint('yolov8s', checkpoint,
            None if stage == 'supervised' else source,
            None if stage == 'supervised' else seed,
            None if stage == 'supervised' else stage)
        manifest['inputs'].append(record)
        dump(output / 'validation_manifest.json', manifest)
        cache_name = f'validation_predictions_{stage}_seed{seed}.pt'
        cached = reuse/cache_name if reuse else None
        adapter = None
        if cached and cached.is_file():
            previous = json.loads((reuse/'validation_manifest.json').read_text())
            if previous['device'] != device or previous['dataset'] != dataset or previous['protocol'] != manifest['protocol']:
                raise ValueError('Prediction cache protocol/device/dataset mismatch')
            saved = torch.load(cached, map_location='cpu', weights_only=False)
            if saved['checkpoint_sha256'] != record['checkpoint_sha256']:
                raise ValueError('Prediction cache checkpoint mismatch')
            preds, targets = saved['predictions'], saved['targets']
            record['reused_prediction_cache'] = {'path': str(cached), 'sha256': sha(cached)}
            print(f'REUSE seed={seed} stage={stage}', flush=True)
        else:
            adapter = load_adapter('yolov8s', str(source), device)
            if payload:
                adapter.load_state_dict(payload['state_dict'], strict=True)
            loader = get_pest_dataloader(DATA_ROOT, 'val', batch_size=8, num_workers=0, seed=0)
            preds, targets = [], []
            print(f'VALIDATION seed={seed} stage={stage}', flush=True)
            with torch.no_grad():
                for batch, (images, gt) in enumerate(loader):
                    result = adapter.forward_with_grad(images.to(device), conf_thres=.001, iou_thres=.60)
                    preds.extend({k: v[k].detach().cpu() for k in ('boxes', 'labels', 'scores')} for v in result)
                    targets.extend(gt)
                    if (batch+1) % 15 == 0:
                        print(f'  {len(preds)}/{len(loader.dataset)} images', flush=True)
        del payload
        if len(preds) != dataset['counts']['valid']['images'] or len(targets) != len(preds):
            raise ValueError('Partial validation')
        metric = MeanAveragePrecision(class_metrics=False, max_detection_thresholds=[1, 10, 300],
                                      backend='faster_coco_eval')
        metric.update(preds, targets)
        ap = metric.compute()
        counts = _operating_point_counts(preds, targets)
        p, r, f, macro = _summarize_counts(counts)
        tp, fp, fn = [sum(v[i] for v in counts.values()) for i in range(3)]
        if baseline_tp is None:
            baseline_tp = tp
        points = ranked_points(preds, targets)
        # Cross-check the curve against the canonical implementation at fixed thresholds.
        for threshold in (.05, .1, .25, .5, .75):
            direct = _operating_point_counts(preds, targets, threshold)
            expected_counts = [sum(v[i] for v in direct.values()) for i in range(3)]
            eligible = [v for v in points if v['threshold'] >= threshold]
            selected = eligible[-1] if eligible else {'TP': 0, 'FP': 0, 'FN': tp+fn}
            if expected_counts != [selected[k] for k in ('TP', 'FP', 'FN')]:
                raise ValueError(f'Ranked curve mismatch at {threshold}')
        feasible = [v for v in points if v['TP'] >= baseline_tp]
        best = max(feasible, key=lambda v: (v['precision'], v['recall'], v['threshold']))
        entry = {'seed': seed, 'stage': stage, 'n_images': len(preds),
                 'mAP50_95': float(ap['map']), 'precision_at_025': p, 'recall_at_025': r,
                 'f1_at_025': f, 'macro_f1_at_025': macro, 'TP': tp, 'FP': fp, 'FN': fn,
                 'reference_S0_TP': baseline_tp,
                 **{f'matched_recall_{k}': v for k, v in best.items()}}
        expected_ap = json.loads((checkpoint.parent/'completed.json').read_text())['last']['mAP50_95'] if stage != 'supervised' else None
        if expected_ap is not None:
            entry.update(historical_cuda_mAP50_95=expected_ap,
                         backend_delta_mAP_pp=(float(ap['map'])-expected_ap)*100,
                         historical_parity_pass=abs(float(ap['map'])-expected_ap) <= 1e-4)
        else:
            entry.update(historical_cuda_mAP50_95='', backend_delta_mAP_pp='', historical_parity_pass='')
        evaluations.append(entry)
        all_points.extend({'seed': seed, 'stage': stage, **v} for v in points)
        torch.save({'predictions': preds, 'targets': targets,
                    'checkpoint_sha256': record['checkpoint_sha256']},
                   output / cache_name)
        write_csv(output / 'validation_precision.csv', evaluations)
        print('DONE', json.dumps(entry), flush=True)
        if expected_ap is not None and not entry['historical_parity_pass'] and not allow_backend_difference:
            raise ValueError(f'Backend parity mismatch: fresh={float(ap["map"])} historical={expected_ap}')
        if sha(checkpoint) != record['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during validation')
        del adapter, preds, targets, metric
    if dataset_manifest() != dataset:
        raise ValueError('Dataset changed during validation')
    write_csv(output / 'validation_pr_curve.csv', all_points)
    dump(output / 'validation_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--infer-validation', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, choices=(42, 43, 44), default=[42, 43, 44])
    parser.add_argument('--reuse-predictions', type=Path, help='Reuse hash-identified fresh validation predictions with identical protocol')
    parser.add_argument('--allow-backend-difference', action='store_true', help='Record historical CUDA parity differences; compare only within a separate CPU cohort')
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error('Duplicate seeds')
    if args.allow_backend_difference and args.device != 'cpu':
        parser.error('Backend-difference diagnostic is restricted to a separate CPU cohort')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'status.json', {'status': 'running'})
    try:
        saved_results(output)
        if args.infer_validation:
            validation(output, args.seeds, args.device,
                       args.reuse_predictions.resolve() if args.reuse_predictions else None,
                       args.allow_backend_difference)
        dump(output / 'completed.json', {'status': 'complete', 'test_inference': False,
             'validation_inference': args.infer_validation, 'seeds': args.seeds,
             'script_sha256': sha(Path(__file__)),
             'saved_results_sha256': sha(RUN/'results_all.csv')})
        dump(output / 'status.json', {'status': 'complete'})
    except Exception as exc:
        dump(output / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
