"""
train_supervised.py – Giai đoạn 1: Supervised training cho tất cả YOLO models.

Chạy tuần tự hoặc chọn một model cụ thể:
    python train_supervised.py                    # train tất cả
    python train_supervised.py --model yolov8n    # chỉ train YOLOv8n
    python train_supervised.py --model dp_yolo    # chỉ train DP-YOLO

Sau khi chạy xong, checkpoints được lưu tại:
    checkpoints/<model_name>/weights/best.pt
"""

import argparse
import subprocess
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình model
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    'yolov5s': {
        'framework': 'v5',
        'weights':   'yolov5s.pt',
        'cfg':       None,           # dùng cấu hình mặc định
    },
    'yolov8n': {
        'framework': 'ultralytics',
        'weights':   'yolov8n.pt',
    },
    'yolov8s': {
        'framework': 'ultralytics',
        'weights':   'yolov8s.pt',
    },
    'yolov11n': {
        'framework': 'ultralytics',
        'weights':   'yolo11n.pt',
    },
    'yolov11s': {
        'framework': 'ultralytics',
        'weights':   'yolo11s.pt',
    },
    'dp_yolo': {
        'framework': 'v5',
        'weights':   'yolov5s.pt',
        'cfg':       'models/dp_yolo/dp_yolo.yaml',
    },
}

# Hyperparameters chung
COMMON = {
    'data':       'configs/pest.yaml',
    'imgsz':      640,
    'epochs':     300,
    'batch':      32,
    'workers':    8,
    'device':     '0',       # GPU id, hoặc 'cpu'
    'patience':   50,        # early stopping
    'project':    'checkpoints',
    'exist_ok':   True,      # không báo lỗi nếu thư mục đã tồn tại
}


# ─────────────────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────────────────

def train_yolov5(name: str, cfg: dict):
    """
    Train YOLOv5 / DP-YOLO.

    - DP-YOLO (cfg có 'cfg' key): dùng dp_yolo_train.py – áp dụng
      W3F_MPDIoU loss, PSA label assignment, và custom modules.
    - YOLOv5s standard: dùng yolov5/train.py trực tiếp.
    """
    is_dp_yolo = bool(cfg.get('cfg'))   # chỉ dp_yolo mới có --cfg flag

    if is_dp_yolo:
        # Dùng wrapper để đảm bảo patches (loss, PSA, modules) được apply
        # TRƯỚC KHI YOLOv5 training bắt đầu
        script = 'dp_yolo_train.py'
    else:
        script = 'yolov5/train.py'

    cmd = [
        'python', script,
        f"--weights={cfg['weights']}",
        f"--data={COMMON['data']}",
        f"--imgsz={COMMON['imgsz']}",
        f"--epochs={COMMON['epochs']}",
        f"--batch-size={COMMON['batch']}",
        f"--workers={COMMON['workers']}",
        f"--device={COMMON['device']}",
        f"--project={COMMON['project']}",
        f"--name={name}",
        '--optimizer=SGD',
        '--hyp=configs/hyp.pest.yaml',
        '--exist-ok',
        '--save-period=50',   # lưu checkpoint mỗi 50 epoch
    ]
    if cfg.get('cfg'):
        cmd.append(f"--cfg={cfg['cfg']}")

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def train_ultralytics(name: str, cfg: dict):
    """Train YOLOv8 / YOLOv11 qua Ultralytics Python API."""
    from ultralytics import YOLO

    model = YOLO(cfg['weights'])
    model.train(
        data=COMMON['data'],
        imgsz=COMMON['imgsz'],
        epochs=COMMON['epochs'],
        batch=COMMON['batch'],
        workers=COMMON['workers'],
        device=COMMON['device'],
        patience=COMMON['patience'],
        project=COMMON['project'],
        name=name,
        exist_ok=COMMON['exist_ok'],
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.3,
        mosaic=1.0,
        mixup=0.1,
        save_period=50,
        plots=True,
    )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Supervised training – Giai đoạn 1')
    parser.add_argument(
        '--model', default='all',
        choices=['all'] + list(MODELS.keys()),
        help='Model cần train (mặc định: all)',
    )
    args = parser.parse_args()

    targets = MODELS if args.model == 'all' else {args.model: MODELS[args.model]}

    results = {}
    for name, cfg in targets.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name}  (framework: {cfg['framework']})")
        print(f"{'='*60}")
        try:
            if cfg['framework'] == 'v5':
                ok = train_yolov5(name, cfg)
            else:
                ok = train_ultralytics(name, cfg)
            results[name] = 'OK' if ok else 'FAILED'
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = f'ERROR: {e}'

    print(f"\n{'='*60}")
    print('  SUMMARY')
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name:15s}  {status}")


if __name__ == '__main__':
    main()
