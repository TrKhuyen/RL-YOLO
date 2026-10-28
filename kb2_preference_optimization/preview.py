"""Human inspection of GT/chosen/rejected; never used as policy input."""
from pathlib import Path

import cv2
import numpy as np


def save_preview(path, image, target, record):
    image = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    panels = []
    for side in ('GT', 'chosen', 'rejected'):
        panel = image.copy()
        prediction = target if side == 'GT' else record[f'{side}_detection']
        color = (60, 220, 60) if side != 'rejected' else (60, 80, 230)
        for box, label in zip(prediction['boxes'], prediction['labels']):
            x1, y1, x2, y2 = (int(float(v)) for v in box)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, 1)
            cv2.putText(panel, str(int(label)), (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
        panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        title = side if side == 'GT' else f'{side} Q={record[f"{side}_quality"]["score"]:.3f}'
        cv2.putText(panel, title, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1)
        panels.append(panel)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.concatenate(panels, axis=1)):
        raise OSError(f'Could not save preference preview: {path}')
