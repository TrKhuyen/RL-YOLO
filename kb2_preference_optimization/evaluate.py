"""
evaluate.py – Đánh giá và so sánh: Supervised vs RL Level 1/2/3.

So sánh:
  - Supervised-only checkpoint (baseline)
  - RL Level 1 (REINFORCE + EMA)
  - RL Level 2 (GRPO-style Group Aug)
  - RL Level 3 (GRPO + DAPO)

Metrics: mAP@50, mAP@50-95, AP_small, Recall, FPS.

Chạy:
    cd kb2_preference_optimization
    python evaluate.py
    python evaluate.py --model dp_yolo --split test
    python evaluate.py --levels 1 2 3  # so sánh cả 3 levels
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torchmetrics.detection import MeanAveragePrecision

from dataloader import get_pest_dataloader


# =============================================================================
# Cấu hình experiments
# =============================================================================

# Supervised checkpoints (từ kb1_reward_guided_training hoặc train riêng)
SUPERVISED = {
    'YOLOv5s':  'checkpoints/yolov5s/weights/best.pt',
    'YOLOv8n':  'checkpoints/yolov8n/weights/best.pt',
    'YOLOv8s':  'checkpoints/yolov8s/weights/best.pt',
    'YOLOv11n': 'checkpoints/yolov11n/weights/best.pt',
    'YOLOv11s': 'checkpoints/yolov11s/weights/best.pt',
    'DP-YOLO':  'checkpoints/dp_yolo/weights/best.pt',
}

# RL checkpoints (từ train_rl.py – mỗi level có 1 best.pt)
RL_CHECKPOINTS = {
    'YOLOv5s': {
        'l1': 'rl_checkpoints/yolov5s_rl_l1_best.pt',
        'l2': 'rl_checkpoints/yolov5s_rl_l2_best.pt',
        'l3': 'rl_checkpoints/yolov5s_rl_l3_best.pt',
    },
    'YOLOv8n': {
        'l1': 'rl_checkpoints/yolov8n_rl_l1_best.pt',
        'l2': 'rl_checkpoints/yolov8n_rl_l2_best.pt',
        'l3': 'rl_checkpoints/yolov8n_rl_l3_best.pt',
    },
    'YOLOv8s': {
        'l1': 'rl_checkpoints/yolov8s_rl_l1_best.pt',
        'l2': 'rl_checkpoints/yolov8s_rl_l2_best.pt',
        'l3': 'rl_checkpoints/yolov8s_rl_l3_best.pt',
    },
    'YOLOv11n': {
        'l1': 'rl_checkpoints/yolov11n_rl_l1_best.pt',
        'l2': 'rl_checkpoints/yolov11n_rl_l2_best.pt',
        'l3': 'rl_checkpoints/yolov11n_rl_l3_best.pt',
    },
    'YOLOv11s': {
        'l1': 'rl_checkpoints/yolov11s_rl_l1_best.pt',
        'l2': 'rl_checkpoints/yolov11s_rl_l2_best.pt',
        'l3': 'rl_checkpoints/yolov11s_rl_l3_best.pt',
    },
    'DP-YOLO': {
        'l1': 'rl_checkpoints/dp_yolo_rl_l1_best.pt',
        'l2': 'rl_checkpoints/dp_yolo_rl_l2_best.pt',
        'l3': 'rl_checkpoints/dp_yolo_rl_l3_best.pt',
    },
}

# Framework mapping
FRAMEWORKS = {
    'YOLOv5s':  'v5',
    'YOLOv8n':  'ultralytics',
    'YOLOv8s':  'ultralytics',
    'YOLOv11n': 'ultralytics',
    'YOLOv11s': 'ultralytics',
    'DP-YOLO':  'v5',
}


# =============================================================================
# Load model (xử lý cả supervised và RL checkpoint)
# =============================================================================

def _load_model_for_eval(
    checkpoint:      str,
    framework:       str,
    device:          str,
    conf_thres:      float = 0.25,
    iou_thres:       float = 0.45,
    supervised_ckpt: str   = None,
):
    """
    Load model về dạng callable để evaluate.
    Xử lý 2 loại checkpoint:
      1. Standard (supervised): ultralytics/torch format
      2. RL checkpoint (dict với is_rl_checkpoint=True)

    Returns:
        callable(images) → list[dict(boxes, scores, labels)]
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        return None

    # Thử detect RL checkpoint
    try:
        data = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        is_rl = isinstance(data, dict) and data.get('is_rl_checkpoint', False)
    except Exception:
        is_rl = False

    if is_rl:
        # RL checkpoint: cần load architecture từ supervised ckpt trước
        assert supervised_ckpt and Path(supervised_ckpt).exists(), (
            f"RL checkpoint cần supervised_ckpt để load architecture: {supervised_ckpt}"
        )
        model = _load_standard(supervised_ckpt, framework, device)
        # Apply RL weights
        state = data.get('state_dict', data)
        if hasattr(model, 'model'):
            model.model.load_state_dict(state, strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"    RL checkpoint loaded (step={data.get('step', '?')}, "
              f"level={data.get('level', '?')})")
    else:
        model = _load_standard(str(ckpt_path), framework, device)

    model.eval() if hasattr(model, 'eval') else None
    return model


def _load_standard(checkpoint: str, framework: str, device: str):
    """Load standard (non-RL) checkpoint."""
    if framework == 'v5':
        import sys
        sys.path.insert(0, 'yolov5')
        from models.common import DetectMultiBackend
        return DetectMultiBackend(checkpoint, device=torch.device(device))
    else:
        from ultralytics import YOLO
        return YOLO(checkpoint).to(device)


# =============================================================================
# Evaluate single checkpoint
# =============================================================================

def evaluate_checkpoint(
    model,
    val_loader,
    framework:  str,
    device:     str   = 'cuda',
    conf_thres: float = 0.25,
    iou_thres:  float = 0.45,
) -> dict:
    """
    Evaluate 1 model checkpoint trên val/test set.
    Trả về dict: mAP50, mAP50-95, AP_small, Recall, FPS.
    """
    metric = MeanAveragePrecision(
        iou_thresholds=[0.5, 0.75],
        extended_summary=True,
        class_metrics=False,
    )

    preds_all, targets_all = [], []
    t0 = time.time()
    n_imgs = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)

            if framework == 'ultralytics':
                results = model(images, conf=conf_thres, iou=iou_thres, verbose=False)
                for r, t in zip(results, targets):
                    preds_all.append({
                        'boxes':  r.boxes.xyxy.cpu(),
                        'scores': r.boxes.conf.cpu(),
                        'labels': r.boxes.cls.int().cpu(),
                    })
                    targets_all.append({
                        'boxes':  t['boxes'].cpu(),
                        'labels': t['labels'].int().cpu(),
                    })
            else:
                # YOLOv5 style
                from utils.general import non_max_suppression
                out = model.model(images)
                if isinstance(out, tuple):
                    out = out[0]
                dets = non_max_suppression(out.detach(), conf_thres, iou_thres)
                for det, t in zip(dets, targets):
                    if det is not None and len(det):
                        preds_all.append({
                            'boxes':  det[:, :4].cpu(),
                            'scores': det[:, 4].cpu(),
                            'labels': det[:, 5].int().cpu(),
                        })
                    else:
                        preds_all.append({
                            'boxes':  torch.zeros((0, 4)),
                            'scores': torch.zeros(0),
                            'labels': torch.zeros(0, dtype=torch.int),
                        })
                    targets_all.append({
                        'boxes':  t['boxes'].cpu(),
                        'labels': t['labels'].int().cpu(),
                    })

            n_imgs += len(images)

    elapsed = time.time() - t0
    fps = n_imgs / elapsed

    metric.update(preds_all, targets_all)
    res = metric.compute()

    return {
        'mAP50':    float(res.get('map_50',    torch.tensor(0.0)).item()),
        'mAP50-95': float(res.get('map',       torch.tensor(0.0)).item()),
        'AP_small': float(res.get('map_small', torch.tensor(0.0)).item()),
        'Recall':   float(res.get('mar_100',   torch.tensor(0.0)).item()),
        'FPS':      fps,
    }


# =============================================================================
# Run full comparison
# =============================================================================

def run_comparison(args):
    """So sánh supervised vs RL Level 1/2/3 cho tất cả model."""
    device = args.device
    models_to_eval = (
        list(SUPERVISED.keys()) if args.model == 'all'
        else [args.model]
    )
    levels_to_eval = args.levels

    data_root = args.data_root
    split     = args.split
    loader    = get_pest_dataloader(
        data_root, split=split, batch_size=16, img_size=640
    )

    rows = []

    for model_name in models_to_eval:
        framework = FRAMEWORKS.get(model_name, 'ultralytics')
        sup_ckpt  = SUPERVISED.get(model_name, '')
        rl_ckpts  = RL_CHECKPOINTS.get(model_name, {})

        # ── Supervised baseline ─────────────────────────────────────────
        if Path(sup_ckpt).exists():
            print(f'Evaluating {model_name} [supervised]...')
            model = _load_model_for_eval(sup_ckpt, framework, device)
            if model:
                m = evaluate_checkpoint(model, loader, framework, device)
                rows.append({'Model': model_name, 'Stage': 'supervised', **m})
                print(f'  mAP50={m["mAP50"]:.4f} AP_small={m["AP_small"]:.4f} '
                      f'Recall={m["Recall"]:.4f} FPS={m["FPS"]:.1f}')
        else:
            print(f'  SKIP {model_name} [supervised]: not found at {sup_ckpt}')

        # ── RL levels ───────────────────────────────────────────────────
        for level_key in [f'l{l}' for l in levels_to_eval]:
            rl_ckpt = rl_ckpts.get(level_key, '')
            if not rl_ckpt or not Path(rl_ckpt).exists():
                print(f'  SKIP {model_name} [{level_key}]: not found at {rl_ckpt}')
                continue

            print(f'Evaluating {model_name} [RL {level_key}]...')
            model = _load_model_for_eval(
                rl_ckpt, framework, device, supervised_ckpt=sup_ckpt
            )
            if model:
                m = evaluate_checkpoint(model, loader, framework, device)
                rows.append({'Model': model_name, 'Stage': f'RL {level_key}', **m})
                print(f'  mAP50={m["mAP50"]:.4f} AP_small={m["AP_small"]:.4f} '
                      f'Recall={m["Recall"]:.4f} FPS={m["FPS"]:.1f}')

    if not rows:
        print('\nNo results. Run train_rl.py first.')
        return None

    df = pd.DataFrame(rows)

    # ── Tính delta RL - Supervised ──────────────────────────────────────
    print(f'\n{"="*70}')
    print('COMPARISON: RL vs Supervised')
    print(f'{"="*70}')

    for model_name in df['Model'].unique():
        df_m = df[df['Model'] == model_name]
        sup_row = df_m[df_m['Stage'] == 'supervised']
        if sup_row.empty:
            continue
        print(f'\n{model_name}:')
        for _, rl_row in df_m[df_m['Stage'].str.startswith('RL')].iterrows():
            stage = rl_row['Stage']
            for metric in ['mAP50', 'AP_small', 'Recall']:
                sup_val = sup_row[metric].values[0]
                rl_val  = rl_row[metric]
                delta   = rl_val - sup_val
                sign    = '+' if delta >= 0 else ''
                print(f'  {stage} | {metric}: {sup_val:.4f} → {rl_val:.4f} '
                      f'({sign}{delta:.4f})')

    # ── Lưu kết quả ─────────────────────────────────────────────────────
    Path('results/tables').mkdir(parents=True, exist_ok=True)
    out_csv = f'results/tables/comparison_{split}.csv'
    df.to_csv(out_csv, index=False)

    print(f'\n{"="*70}')
    print(f'Full table ({split}):')
    print(df.to_markdown(index=False, floatfmt='.4f'))
    print(f'\nSaved: {out_csv}')
    return df


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate RL Fine-tuning Comparison')
    parser.add_argument('--model', default='all',
                        choices=['all', 'YOLOv5s', 'YOLOv8n', 'YOLOv8s',
                                 'YOLOv11n', 'YOLOv11s', 'DP-YOLO'])
    parser.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3],
                        help='RL levels to compare (e.g. --levels 2 3)')
    parser.add_argument('--split',  default='val', choices=['val', 'test'])
    parser.add_argument('--data-root', default='../pre-data/data/v2i')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    run_comparison(args)


if __name__ == '__main__':
    main()
