"""Compare verified supervised best.pt checkpoints with the canonical evaluator."""
import argparse
import gc
import json
import os
from pathlib import Path

import pandas as pd
import torch
import ultralytics
import yaml

from canonical_eval import evaluate_adapter
from dataloader import get_pest_dataloader
from run_provenance import dataset_manifest, verify_supervised
from train_rl import CHECKPOINTS, load_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(CHECKPOINTS),
                        default=list(CHECKPOINTS))
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('batch-size must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Use a fresh evaluation directory: {args.output}')
    dataset = dataset_manifest()
    manifests = {name: verify_supervised(name, CHECKPOINTS[name], dataset)
                 for name in args.models}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        'dataset': dataset, 'checkpoints': manifests,
        'device': args.device, 'batch_size': args.batch_size,
        'torch': torch.__version__, 'ultralytics': ultralytics.__version__,
        'gpu': torch.cuda.get_device_name(0) if str(args.device).startswith('cuda') else None,
        'protocol': {'imgsz': 640, 'conf': .001, 'nms_iou': .60, 'max_det': 300,
                     'operating_conf': .25, 'operating_iou': .5,
                     'yolo26_head': 'one-to-many', 'dtype': 'FP32',
                     'latency': 'batched forward + canonical NMS; first batch excluded; no I/O'},
    }
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    rows = []
    for name in args.models:
        flags = manifests[name]['dp_flags']
        os.environ['DP_YOLO_USE_W3F'] = '1' if flags['w3f'] else '0'
        os.environ['DP_YOLO_USE_PSA'] = '1' if flags['psa'] else '0'
        print(f'Loading {name}', flush=True)
        adapter = load_adapter(name, str(CHECKPOINTS[name]), args.device)
        params = sum(p.numel() for p in adapter.parameters())
        for split in args.splits:
            loader = get_pest_dataloader(dataset['root'], split=split,
                                        batch_size=args.batch_size, img_size=640,
                                        num_workers=0, shuffle=False)
            metrics = evaluate_adapter(adapter, loader, device=args.device,
                                       class_metrics=True, measure_latency=True)
            row = {'Model': name, 'Split': split, 'Parameters': params,
                   'Checkpoint_MB': CHECKPOINTS[name].stat().st_size / 1024**2,
                   **metrics}
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.output / 'results.csv', index=False)
            print(f"DONE {name}/{split}: mAP50={metrics['mAP50']:.6f} "
                  f"mAP50_95={metrics['mAP50_95']:.6f} APs={metrics['APs']:.6f} "
                  f"P={metrics['precision']:.6f} R={metrics['operating_recall']:.6f} "
                  f"FPS={metrics['fps']:.1f}", flush=True)
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    names = yaml.safe_load((Path(__file__).resolve().parent / 'configs/pest.yaml').read_text(encoding='utf-8'))['names']
    per_class = []
    for row in rows:
        for cls, ap, ar in zip(row['classes'], row['map_per_class'], row['mar_per_class']):
            per_class.append({'Model': row['Model'], 'Split': row['Split'], 'Class_ID': cls,
                              'Class': names[cls], 'AP50_95': ap, 'AR300': ar})
    pd.DataFrame(per_class).to_csv(args.output / 'per_class.csv', index=False)
    if dataset_manifest() != dataset:
        raise RuntimeError('Dataset changed during evaluation')
    (args.output / 'completed.json').write_text(json.dumps({'rows': len(rows), 'models': args.models,
                                                          'splits': args.splits}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
