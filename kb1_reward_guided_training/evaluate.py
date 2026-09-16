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
        'supervised': 'checkpoints/yolov5s/weights/best.pt',
        'rl':         'rl_checkpoints/yolov5s_rl_best.pt',
        'framework':  'v5',
    },
    'YOLOv8n': {
        'supervised': 'checkpoints/yolov8n/weights/best.pt',
        'rl':         'rl_checkpoints/yolov8n_rl_best.pt',
        'framework':  'ultralytics',
    },
    'YOLOv8s': {
        'supervised': 'checkpoints/yolov8s/weights/best.pt',
        'rl':         'rl_checkpoints/yolov8s_rl_best.pt',
        'framework':  'ultralytics',
    },
    'YOLOv11n': {
        'supervised': 'checkpoints/yolov11n/weights/best.pt',
        'rl':         'rl_checkpoints/yolov11n_rl_best.pt',
        'framework':  'ultralytics',
    },
    'YOLOv11s': {
        'supervised': 'checkpoints/yolov11s/weights/best.pt',
        'rl':         'rl_checkpoints/yolov11s_rl_best.pt',
        'framework':  'ultralytics',
    },
    'DP-YOLO': {
        'supervised': 'checkpoints/dp_yolo/weights/best.pt',
        'rl':         'rl_checkpoints/dp_yolo_rl_best.pt',
        'framework':  'v5',
    },
}

for _paths in EXPERIMENTS.values():
    for _stage in ('supervised', 'rl'):
        _paths[_stage] = str(PROJECT_ROOT / _paths[_stage])

_legacy_v8n = PROJECT_ROOT.parent / 'runs' / 'detect' / 'checkpoints' / 'yolov8n' / 'weights' / 'best.pt'
if not Path(EXPERIMENTS['YOLOv8n']['supervised']).exists() and _legacy_v8n.exists():
    EXPERIMENTS['YOLOv8n']['supervised'] = str(_legacy_v8n)


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
    from train_rl import load_adapter
    is_rl = base_checkpoint is not None
    data = (torch.load(checkpoint, map_location='cpu', weights_only=False)
            if is_rl else {})
    source = base_checkpoint if is_rl else checkpoint
    if is_rl and not source:
        raise ValueError('RL checkpoint requires base_checkpoint')
    adapter = load_adapter(model_name, str(source), device)
    if is_rl:
        adapter.load_state_dict(data['state_dict'], strict=True)
    adapter.eval_mode()
    return adapter, data if is_rl else {}


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
    metrics['seed'] = metadata.get('seed')
    return metrics


def evaluate_model(checkpoint, dataloader, framework='ultralytics',
                   device='cuda', conf_thres=0.001, iou_thres=0.60,
                   supervised_ckpt=None, model_name=None, max_det=300):
    path = str(supervised_ckpt or checkpoint).lower()
    names = ('dp_yolo', 'yolov11n', 'yolov11s', 'yolov8n', 'yolov8s', 'yolov5s')
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

def run_comparison(
    model_filter: str  = 'all',
    split:        str  = 'val',
    device:       str  = 'cuda',
    seed:         int  = 42,
) -> pd.DataFrame:
    """
    Chay evaluation toan bo, tinh delta RL vs supervised, luu CSV.
    """
    loader = get_pest_dataloader(
        DATA_ROOT, split=split, batch_size=16, num_workers=4,
    )

    targets_to_eval = (
        EXPERIMENTS if model_filter == 'all'
        else {model_filter: EXPERIMENTS[model_filter]}
    )

    rows = []
    for model_name, paths in targets_to_eval.items():
        adapter_name = model_name.lower().replace('-', '_')
        stage_paths = {
            'supervised': paths['supervised'],
            'native_only': str(
                PROJECT_ROOT / 'rl_checkpoints'
                / f'{adapter_name}_seed{seed}_native_only_best.pt'
            ),
            'kb1b': str(
                PROJECT_ROOT / 'rl_checkpoints'
                / f'{adapter_name}_seed{seed}_rl_best.pt'
            ),
        }
        for stage, ckpt in stage_paths.items():
            if not ckpt or not Path(ckpt).exists():
                print(f"  SKIP {model_name} [{stage}]: {ckpt}")
                continue

            print(f"  Evaluating {model_name} [{stage}]...")
            try:
                metrics = canonical_evaluate_model(
                    adapter_name, ckpt, loader,
                    device=device,
                    supervised_ckpt=(
                        paths['supervised'] if stage != 'supervised' else None
                    ),
                )
                rows.append({
                    'Model': model_name,
                    'Stage': stage,
                    **metrics,
                })
                print(f"    mAP50={metrics['mAP50']:.4f}  "
                      f"APs={metrics['APs']:.4f}  "
                      f"recall={metrics['recall']:.4f}  "
                      f"fps={metrics['fps']:.1f}")
            except Exception as e:
                print(f"    ERROR: {e}")

    if not rows:
        print("Không có kết quả nào. Kiểm tra checkpoints.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Tính delta RL vs supervised ─────────────────────────────────────
    metrics_cols = ['mAP50', 'mAP50_95', 'APs', 'APm', 'recall']
    df_sup = df[df['Stage'] == 'supervised'].set_index('Model')[metrics_cols]
    df_native = df[df['Stage'] == 'native_only'].set_index('Model')[metrics_cols]
    df_kb1b = df[df['Stage'] == 'kb1b'].set_index('Model')[metrics_cols]
    delta_sup = (df_kb1b - df_sup).add_suffix('_kb1b_minus_supervised')
    delta_native = (
        df_kb1b - df_native
    ).add_suffix('_kb1b_minus_native_only')
    df_delta = delta_sup.join(delta_native, how='outer').reset_index()

    # ── Lưu kết quả ─────────────────────────────────────────────────────
    out_dir = PROJECT_ROOT / 'results' / 'canonical' / split
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / 'results_full.csv', index=False)
    df_delta.to_csv(out_dir / 'results_delta.csv', index=False)

    # ── In bảng markdown ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print('  KẾT QUẢ SO SÁNH SUPERVISED vs. RL FINE-TUNED')
    print(f"{'='*80}")
    print(df.to_markdown(index=False, floatfmt='.4f'))

    print(f"\n{'='*80}")
    print('  DELTA GHÉP CẶP (dương = KB1-B tốt hơn)')
    print(f"{'='*80}")
    print(df_delta.to_markdown(index=False, floatfmt='+.4f'))

    print(f"\n  → Đã lưu: {out_dir}/results_full.csv")
    print(f"  → Đã lưu: {out_dir}/results_delta.csv")

    return df


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
    args = parser.parse_args()

    run_comparison(
        model_filter=args.model,
        split=args.split,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
