"""
evaluate.py – Giai đoạn 3: Đánh giá và so sánh toàn bộ model.

So sánh supervised-only vs. RL-finetuned cho từng model.
Metrics: mAP@50, mAP@50-95, AP_small, Recall, FPS.

Chạy:
    python evaluate.py                   # evaluate tất cả
    python evaluate.py --model dp_yolo   # chỉ evaluate DP-YOLO
    python evaluate.py --split test      # dùng tập test thay vì val
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO

from dataloader import get_pest_dataloader

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT.parent / 'pre-data' / 'data' / 'v2i_cleanned'


# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình experiments
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    'YOLOv5s': {
        'supervised': 'checkpoint_based/yolov5s/weights/best.pt',
        'rl':         'checkpoint_reward_guide_trainning/yolov5s_seed42_rl_best.pt',
        'framework':  'v5',
    },
    'YOLOv8n': {
        'supervised': 'checkpoint_based/yolov8n/weights/best.pt',
        'rl':         'checkpoint_reward_guide_trainning/yolov8n_seed42_rl_best.pt',
        'framework':  'ultralytics',
    },
    'YOLOv8s': {
        'supervised': 'checkpoint_based/yolov8s/weights/best.pt',
        'rl':         'checkpoint_reward_guide_trainning/yolov8s_seed42_rl_best.pt',
        'framework':  'ultralytics',
    },
    'YOLOv11n': {
        'supervised': 'checkpoint_based/yolov11n/weights/best.pt',
        'rl':         'checkpoint_reward_guide_trainning/yolov11n_seed42_rl_best.pt',
        'framework':  'ultralytics',
    },
    'YOLOv11s': {
        'supervised': 'checkpoint_based/yolov11s/weights/best.pt',
        'rl':         'checkpoint_reward_guide_trainning/yolov11s_seed42_rl_best.pt',
        'framework':  'ultralytics',
    },
    'YOLO26n': {
        'supervised': 'checkpoint_based/yolo26n/weights/best.pt',
        'rl':         'checkpoint_reward_guide_trainning/yolo26n_seed42_rl_best.pt',
        'framework':  'ultralytics',
    },
    'DP-YOLO': {
        'supervised': 'checkpoint_based/dp_yolo/weights/best.pt',
        'rl':         'checkpoint_reward_guide_trainning/dp_yolo_seed42_rl_best.pt',
        'framework':  'v5',
    },
}

for _paths in EXPERIMENTS.values():
    for _stage in ('supervised', 'rl'):
        _paths[_stage] = str(PROJECT_ROOT / _paths[_stage])

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load model (xu ly ca supervised va RL checkpoint)
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_for_eval(
    checkpoint:      str,
    framework:       str,
    device:          str,
    conf_thres:      float = 0.001,
    iou_thres:       float = 0.60,
    supervised_ckpt: str | None = None,
):
    """
    Load model cho evaluation, xu ly 2 truong hop:
    1. Standard checkpoint (yolov5 format hoac ultralytics)
    2. RL checkpoint (dict voi is_rl_checkpoint=True, luu boi train_rl.py)

    Voi RL checkpoint, load supervised base truoc, sau do apply RL weights.

    Returns:
        Tuple (model, is_rl: bool)
    """
    # Phat hien dinh dang checkpoint
    if framework == 'v5':
        import sys
        v5_path = str((PROJECT_ROOT / 'yolov5').resolve())
        if v5_path not in sys.path:
            sys.path.insert(0, v5_path)
    try:
        ckpt_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    except ModuleNotFoundError:
        ckpt_data = None
    is_rl = isinstance(ckpt_data, dict) and ckpt_data.get('is_rl_checkpoint', False)

    if framework == 'ultralytics':
        if is_rl:
            if not supervised_ckpt or not Path(supervised_ckpt).exists():
                raise FileNotFoundError(
                    f"RL checkpoint can supervised_ckpt de load model structure. "
                    f"Got: {supervised_ckpt}"
                )
            # Load tu supervised checkpoint, apply RL weights
            model = YOLO(supervised_ckpt)
            state_dict = ckpt_data['state_dict']
            # Ultralytics model: state_dict la model.model.state_dict()
            if hasattr(model.model, 'model'):
                model.model.load_state_dict(state_dict, strict=False)
            else:
                model.model.load_state_dict(state_dict, strict=False)
        else:
            model = YOLO(checkpoint)
        model.overrides['conf'] = conf_thres
        model.overrides['iou']  = iou_thres
        return model, is_rl

    else:  # framework == 'v5'
        if is_rl:
            if not supervised_ckpt or not Path(supervised_ckpt).exists():
                raise FileNotFoundError(
                    f"RL checkpoint can supervised_ckpt. Got: {supervised_ckpt}"
                )
            # Load supervised checkpoint de lay model structure
            sup_data = torch.load(supervised_ckpt, map_location='cpu',
                                  weights_only=False)
            if isinstance(sup_data, dict) and 'model' in sup_data:
                inner_model = sup_data['model'].float().to(device).eval()
            else:
                inner_model = sup_data.float().to(device).eval()

            # Apply RL state_dict
            state_dict = ckpt_data['state_dict']
            inner_model.load_state_dict(state_dict, strict=False)

            # Wrap trong AutoShape de co interface .xyxy
            try:
                from models.common import AutoShape
                model = AutoShape(inner_model)
            except ImportError:
                # Fallback: dung raw model (khong co AutoShape)
                model = inner_model

            model.conf = conf_thres
            model.iou  = iou_thres
            if hasattr(model, 'to'):
                model.to(device)
            model.eval()
            return model, True
        else:
            # Standard YOLOv5 checkpoint
            model = torch.hub.load(
                v5_path, 'custom', path=checkpoint,
                source='local', force_reload=False, verbose=False,
            )
            model.conf = conf_thres
            model.iou  = iou_thres
            model.to(device).eval()
            return model, False


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate single checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _legacy_evaluate_model(
    checkpoint:      str,
    dataloader,
    framework:       str   = 'ultralytics',
    device:          str   = 'cuda',
    conf_thres:      float = 0.001,
    iou_thres:       float = 0.60,
    supervised_ckpt: str | None = None,   # [fix] can thiet khi load RL checkpoint
) -> dict:
    """
    Chay evaluation day du cho 1 checkpoint.
    Xu ly ca supervised checkpoint lan RL checkpoint (dict format).

    Tra ve dict:
        mAP50, mAP50_95, APs (small), APm (medium), recall, fps
    """
    # ── Load model (xu ly ca RL va supervised) ───────────────────────────
    model, _is_rl = _load_model_for_eval(
        checkpoint, framework, device,
        conf_thres=conf_thres, iou_thres=iou_thres,
        supervised_ckpt=supervised_ckpt,
    )

    # ── Metric ───────────────────────────────────────────────────────────
    metric = MeanAveragePrecision(
        iou_thresholds=None,    # dùng COCO standard [0.5:0.95:0.05]
        extended_summary=True,
        class_metrics=True,
        backend='faster_coco_eval',
    )

    all_preds, all_targets = [], []
    t0 = time.perf_counter()
    n_imgs = 0

    with torch.no_grad():
        for images, targets in dataloader:
            # Inference
            if framework == 'ultralytics':
                results = model(images, verbose=False, device=device)
                for r, tgt in zip(results, targets):
                    all_preds.append({
                        'boxes':  r.boxes.xyxy.cpu(),
                        'scores': r.boxes.conf.cpu(),
                        'labels': r.boxes.cls.int().cpu(),
                    })
                    all_targets.append({
                        'boxes':  tgt['boxes'].cpu(),
                        'labels': tgt['labels'].int().cpu(),
                    })
            else:
                # YOLOv5 batch inference
                imgs_np = [
                    (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                    for img in images
                ]
                results = model(imgs_np, size=640)
                for res, tgt in zip(results.xyxy, targets):
                    all_preds.append({
                        'boxes':  res[:, :4].cpu(),
                        'scores': res[:, 4].cpu(),
                        'labels': res[:, 5].int().cpu(),
                    })
                    all_targets.append({
                        'boxes':  tgt['boxes'].cpu(),
                        'labels': tgt['labels'].int().cpu(),
                    })

            n_imgs += len(images)

    elapsed = time.perf_counter() - t0
    fps = n_imgs / elapsed

    # ── Compute metrics ──────────────────────────────────────────────────
    metric.update(all_preds, all_targets)
    res = metric.compute()

    return {
        'mAP50':    res['map_50'].item(),
        'mAP50_95': res['map'].item(),
        'APs':      res.get('map_small',  torch.tensor(0.0)).item(),
        'APm':      res.get('map_medium', torch.tensor(0.0)).item(),
        'recall':   res.get('mar_100',    torch.tensor(0.0)).item(),
        'fps':      fps,
        'n_images': n_imgs,
    }


# Canonical adapter loading. The legacy loader above is retained only for
# backward compatibility with existing imports.
def _load_canonical_adapter(model_name, checkpoint, device, base_checkpoint=None):
    import os
    from train_rl import load_adapter
    from evaluation_provenance import inspect_checkpoint
    record, data = inspect_checkpoint(model_name, checkpoint, base_checkpoint)
    flags = record['source_manifest']['dp_flags']
    os.environ['DP_YOLO_USE_W3F'] = '1' if flags['w3f'] else '0'
    os.environ['DP_YOLO_USE_PSA'] = '1' if flags['psa'] else '0'
    adapter = load_adapter(model_name, record['source'], device)
    if base_checkpoint is not None:
        adapter.load_state_dict(data['state_dict'], strict=True)
    adapter.eval_mode()
    data = dict(data, supervised_seed=record['supervised_seed'])
    return adapter, data


def canonical_evaluate_model(
    model_name, checkpoint, dataloader, device='cuda',
    conf_thres=0.001, iou_thres=0.60, max_det=300,
    supervised_ckpt=None,
):
    from canonical_eval import evaluate_adapter
    adapter, metadata = _load_canonical_adapter(
        model_name, checkpoint, device, supervised_ckpt,
    )
    metrics = evaluate_adapter(
        adapter, dataloader, device=device,
        conf_thres=conf_thres, iou_thres=iou_thres,
        max_det=max_det, class_metrics=True, measure_latency=True,
    )
    metrics['checkpoint_step'] = metadata.get('step', 0)
    metrics['supervised_seed'] = metadata.get('supervised_seed')
    metrics['continuation_seed'] = metadata.get('metadata', {}).get('seed')
    metrics['seed'] = metrics['continuation_seed'] if supervised_ckpt is not None else metrics['supervised_seed']
    return metrics


def evaluate_model(checkpoint, dataloader, framework='ultralytics',
                   device='cuda', conf_thres=0.001, iou_thres=0.60,
                   supervised_ckpt=None, model_name=None, max_det=300):
    path = str(supervised_ckpt or checkpoint).lower()
    names = ('dp_yolo', 'yolov11n', 'yolov11s', 'yolov8n', 'yolov8s', 'yolov5s', 'yolo26n')
    model_name = model_name or next((n for n in names if n in path), None)
    if model_name is None:
        raise ValueError('model_name cannot be inferred')
    return canonical_evaluate_model(
        model_name, checkpoint, dataloader, device,
        conf_thres, iou_thres, max_det, supervised_ckpt)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class breakdown
# ─────────────────────────────────────────────────────────────────────────────

with open(PROJECT_ROOT / 'configs' / 'pest.yaml', encoding='utf-8') as _f:
    CLASS_NAMES = yaml.safe_load(_f)['names']


def evaluate_per_class(
    checkpoint: str,
    dataloader,
    framework: str = 'ultralytics',
    device:    str = 'cuda',
) -> pd.DataFrame:
    """
    Trả về bảng AP per-class để phân tích lớp nào được RL cải thiện nhất.
    """
    if framework == 'ultralytics':
        model = YOLO(checkpoint)
        results = model.val(
            data=str(PROJECT_ROOT / 'configs' / 'pest.yaml'),
            split='val',
            device=device,
            verbose=False,
        )
        ap50 = results.box.ap50      # (nc,)
        rows = [{'class': CLASS_NAMES[i], 'AP50': float(ap50[i])}
                for i in range(len(ap50))]
        return pd.DataFrame(rows)
    else:
        # YOLOv5 val
        import subprocess, json
        r = subprocess.run(
            ['python', 'yolov5/val.py',
             f'--weights={checkpoint}',
             '--data=configs/pest.yaml',
             '--task=val', '--device=0', '--verbose'],
            capture_output=True, text=True, check=True,
        )
        # Tạm thời trả về DataFrame rỗng nếu không parse được
        return pd.DataFrame({'class': CLASS_NAMES,
                             'AP50': [None] * len(CLASS_NAMES)})


# ─────────────────────────────────────────────────────────────────────────────
# Full comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(model_filter='all', split='val', device='cuda', seed=42,
                   checkpoint_source='screening', output=None, screening_root=None,
                   workers=0, batch_size=16, stages=None, selection='best'):
    """Evaluate the entire requested cohort or fail; never overwrite a table."""
    import importlib.metadata
    import json
    import platform
    from datetime import datetime, timezone
    from evaluation_provenance import inspect_checkpoint
    from run_provenance import dataset_manifest, sha256, verify_supervised
    from run_verified import atomic_json
    if checkpoint_source != 'screening':
        raise ValueError('Legacy train_rl artifacts lack verified completed provenance; use screening')
    targets = EXPERIMENTS if model_filter == 'all' else {model_filter: EXPERIMENTS[model_filter]}
    root = Path(screening_root or PROJECT_ROOT / 'checkpoint_reward_guide_trainning'
                / f'screening_seed{seed}_v1').resolve()
    # Check every input before creating output or doing inference.
    stages = tuple(stages or ('supervised', 'native_only', 'kb1b'))
    if 'supervised' not in stages or 'kb1b' not in stages or len(set(stages)) != len(stages):
        raise ValueError('Stages must include supervised and kb1b without duplicates')
    if selection not in ('best', 'last'):
        raise ValueError('Unknown checkpoint selection')
    inputs = []
    for name, paths in targets.items():
        model = name.lower().replace('-', '_')
        source = paths['supervised']
        for stage in stages:
            checkpoint = source if stage == 'supervised' else root / model / stage / f'{selection}.pt'
            record, _ = inspect_checkpoint(model, checkpoint,
                None if stage == 'supervised' else source,
                None if stage == 'supervised' else seed,
                None if stage == 'supervised' else stage)
            inputs.append((name, model, stage, checkpoint, source, record))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = Path(output or PROJECT_ROOT / 'results' / 'canonical_clean'
               / checkpoint_source / f'{split}_{stamp}').resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = dataset_manifest()
    code = [p for p in PROJECT_ROOT.rglob('*.py')
            if p.parent == PROJECT_ROOT or p.parent == PROJECT_ROOT / 'adapters'
            or PROJECT_ROOT / 'models' / 'dp_yolo' in p.parents]
    versions = {}
    for package in ('torch', 'torchvision', 'ultralytics', 'torchmetrics',
                    'faster-coco-eval', 'albumentations', 'numpy', 'opencv-python', 'PyYAML'):
        versions[package] = importlib.metadata.version(package)
    import ultralytics
    dependency_files = {}
    for relative in ('utils/loss.py', 'nn/tasks.py', 'nn/modules/head.py'):
        file = Path(ultralytics.__file__).parent / relative
        dependency_files[relative] = sha256(file)
    manifest = {'created_utc': stamp, 'split': split, 'continuation_seed': seed,
        'screening_root': str(root), 'stages': stages, 'selection': selection, 'expected_rows': len(inputs), 'dataset': dataset,
        'device': device, 'batch_size': batch_size, 'workers': workers,
        'python': platform.python_version(), 'platform': platform.platform(),
        'versions': versions, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(torch.device(device)) if str(device).startswith('cuda') else None,
        'code_sha256': {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in code},
        'ultralytics_code_sha256': dependency_files,
        'protocol': {'imgsz': 640, 'conf': .001, 'nms_iou': .60, 'max_det': 300,
                     'operating_conf': .25, 'operating_iou': .5, 'dtype': 'FP32'},
        'inputs': [item[-1] for item in inputs]}
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out / 'status.json', {'status': 'running'})
    try:
        loader = get_pest_dataloader(DATA_ROOT, split=split, batch_size=batch_size,
                                    num_workers=workers, seed=seed)
        rows = []
        for name, model, stage, checkpoint, source, record in inputs:
            print(f'Evaluating {name}/{stage}', flush=True)
            metrics = canonical_evaluate_model(model, str(checkpoint), loader, device=device,
                supervised_ckpt=None if stage == 'supervised' else source)
            if metrics['n_images'] != len(loader.dataset):
                raise RuntimeError('Incomplete dataset evaluation')
            if sha256(checkpoint) != record['checkpoint_sha256'] or sha256(source) != record['source_sha256']:
                raise RuntimeError('Checkpoint changed during evaluation')
            rows.append({'Model': name, 'Stage': stage, **metrics})
            print(f"DONE {name}/{stage}: mAP50_95={metrics['mAP50_95']:.8f}", flush=True)
        if dataset_manifest() != dataset:
            raise RuntimeError('Dataset changed during evaluation')
        for item in inputs:
            record = item[-1]
            seed_config = record.get('supervised_seed_config')
            if seed_config and sha256(seed_config['path']) != seed_config['sha256']:
                raise RuntimeError('Supervised seed config changed during evaluation')
            if verify_supervised(item[1], item[4], dataset) != record['source_manifest']:
                raise RuntimeError('Supervised provenance changed during evaluation')
            if sha256(item[3]) != record['checkpoint_sha256'] or sha256(item[4]) != record['source_sha256']:
                raise RuntimeError('Evaluation inputs changed')
            if item[2] != 'supervised':
                folder = Path(item[3]).parent
                if sha256(folder / 'manifest.json') != record['run_manifest_sha256'] or sha256(folder / 'completed.json') != record['completed_sha256']:
                    raise RuntimeError('Training provenance changed during evaluation')
        df = pd.DataFrame(rows)
        metrics_cols = ['mAP50', 'mAP50_95', 'APs', 'APm', 'APl', 'AR300',
                        'precision', 'operating_recall', 'f1', 'macro_f1']
        tables = {stage: df[df.Stage == stage].set_index('Model')[metrics_cols] for stage in stages}
        delta = pd.concat([(tables['kb1b'] - tables[stage]).add_suffix('_kb1b_minus_' + stage)
                           for stage in stages if stage != 'kb1b'], axis=1).reset_index()
        for name, table in [('results_full.csv', df), ('results_delta.csv', delta)]:
            tmp = out / (name + '.tmp')
            table.to_csv(tmp, index=False)
            tmp.replace(out / name)
        atomic_json(out / 'completed.json', {'status': 'complete', 'rows': len(rows),
            'manifest_sha256': sha256(out / 'manifest.json'),
            'results_sha256': sha256(out / 'results_full.csv'),
            'delta_sha256': sha256(out / 'results_delta.csv')})
        atomic_json(out / 'status.json', {'status': 'complete'})
        print('Evaluation complete:', out, flush=True)
        return df
    except Exception as exc:
        atomic_json(out / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluation – Giai đoạn 3')
    parser.add_argument('--model',  default='all',
                        choices=['all'] + list(EXPERIMENTS.keys()))
    parser.add_argument('--split',  default='val',
                        choices=['val', 'test'])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-source', choices=['screening', 'train_rl'],
                        default='screening')
    parser.add_argument('--output', type=Path, help='New output directory; existing paths are rejected')
    parser.add_argument('--screening-root', type=Path)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    from experiment_modes import MODES
    parser.add_argument('--stages', nargs='+', choices=('supervised', *MODES))
    parser.add_argument('--selection', choices=('best', 'last'), default='best')
    args = parser.parse_args()
    if args.workers < 0 or args.batch_size <= 0:
        parser.error('workers must be nonnegative; batch-size must be positive')

    run_comparison(
        model_filter=args.model,
        split=args.split,
        device=args.device,
        seed=args.seed,
        checkpoint_source=args.checkpoint_source,
        output=args.output, screening_root=args.screening_root,
        workers=args.workers, batch_size=args.batch_size, stages=args.stages, selection=args.selection,
    )


if __name__ == '__main__':
    main()
