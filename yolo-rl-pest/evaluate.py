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
from torchmetrics.detection import MeanAveragePrecision
from ultralytics import YOLO

from dataloader import get_pest_dataloader


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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate single checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    checkpoint: str,
    dataloader,
    framework:  str   = 'ultralytics',
    device:     str   = 'cuda',
    conf_thres: float = 0.25,
    iou_thres:  float = 0.45,
) -> dict:
    """
    Chạy evaluation đầy đủ cho 1 checkpoint.

    Trả về dict:
        mAP50, mAP50_95, APs (small), APm (medium), recall, fps
    """
    # ── Load model ───────────────────────────────────────────────────────
    if framework == 'ultralytics':
        model = YOLO(checkpoint)
        model.overrides['conf'] = conf_thres
        model.overrides['iou']  = iou_thres
    else:
        # YOLOv5: dùng torch.hub để load và chạy inference
        import sys
        if 'yolov5' not in sys.path:
            sys.path.insert(0, 'yolov5')
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                                path=checkpoint, force_reload=False,
                                verbose=False)
        model.conf = conf_thres
        model.iou  = iou_thres
        model.to(device)
        model.eval()

    # ── Metric ───────────────────────────────────────────────────────────
    metric = MeanAveragePrecision(
        iou_thresholds=None,    # dùng COCO standard [0.5:0.95:0.05]
        extended_summary=True,
        class_metrics=True,
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-class breakdown
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'sau_duc_than', 'sau_cuon_la', 'benh_dom_nau', 'benh_dao_on',
    'benh_kho_van', 'ran_xanh', 'bo_tri', 'ran_nau', 'dom_la', 'healthy',
]


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
            data='configs/pest.yaml',
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
) -> pd.DataFrame:
    """
    Chạy evaluation toàn bộ, tính delta RL vs supervised, lưu CSV.
    """
    loader = get_pest_dataloader(
        'data/pest', split=split, batch_size=16,
    )

    targets_to_eval = (
        EXPERIMENTS if model_filter == 'all'
        else {model_filter: EXPERIMENTS[model_filter]}
    )

    rows = []
    for model_name, paths in targets_to_eval.items():
        for stage in ('supervised', 'rl'):
            ckpt = paths.get(stage)
            if not ckpt or not Path(ckpt).exists():
                print(f"  SKIP {model_name} [{stage}]: {ckpt}")
                continue

            print(f"  Evaluating {model_name} [{stage}]...")
            try:
                metrics = evaluate_model(
                    ckpt, loader,
                    framework=paths['framework'],
                    device=device,
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
    df_rl  = df[df['Stage'] == 'rl'].set_index('Model')[metrics_cols]
    df_delta = (df_rl - df_sup).add_suffix('_delta')
    df_delta = df_delta.reset_index()

    # ── Lưu kết quả ─────────────────────────────────────────────────────
    out_dir = Path('results/tables')
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / 'results_full.csv', index=False)
    df_delta.to_csv(out_dir / 'results_delta.csv', index=False)

    # ── In bảng markdown ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print('  KẾT QUẢ SO SÁNH SUPERVISED vs. RL FINE-TUNED')
    print(f"{'='*80}")
    print(df.to_markdown(index=False, floatfmt='.4f'))

    print(f"\n{'='*80}")
    print('  DELTA: RL - SUPERVISED (dương = RL tốt hơn)')
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
    args = parser.parse_args()

    run_comparison(
        model_filter=args.model,
        split=args.split,
        device=args.device,
    )


if __name__ == '__main__':
    main()
