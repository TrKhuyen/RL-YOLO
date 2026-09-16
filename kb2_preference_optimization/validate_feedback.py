import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from dataloader import PestDataset, get_val_transforms
from feedback import FEEDBACK_TYPES, SCHEMA_VERSION

COLORS = {
    'matched': (40, 180, 40), 'wrong_class': (180, 40, 180),
    'bad_localization': (0, 150, 255), 'false_positive': (0, 0, 255),
    'duplicate': (0, 255, 255), 'missed': (255, 80, 0),
}


def quantiles(values):
    if not values:
        return {}
    data = np.asarray(values, dtype=np.float64)
    return {str(q): round(float(np.quantile(data, q)), 6)
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)}


def analyze(records):
    counts, per_class = Counter(), defaultdict(Counter)
    confidences, ious = defaultdict(list), defaultdict(list)
    for record in records:
        if record.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported feedback schema')
        for kind in FEEDBACK_TYPES:
            for item in record['feedback'][kind]:
                counts[kind] += 1
                per_class[int(item['class_id'])][kind] += 1
                if 'confidence' in item:
                    confidences[kind].append(item['confidence'])
                if 'iou' in item:
                    ious[kind].append(item['iou'])
    return {
        'num_images': len(records),
        'unique_image_ids': len({r['image_id'] for r in records}),
        'feedback_counts': dict(counts),
        'per_class': {str(k): dict(v) for k, v in sorted(per_class.items())},
        'confidence_quantiles': {k: quantiles(v) for k, v in confidences.items()},
        'iou_quantiles': {k: quantiles(v) for k, v in ious.items()},
        'num_preferences': sum(len(r['preferences']) for r in records),
    }


def draw_record(record, dataset, output):
    image, _ = dataset[record['image_id']]
    canvas = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    for kind in FEEDBACK_TYPES:
        for item in record['feedback'][kind]:
            x1, y1, x2, y2 = map(int, item['box'])
            color = COLORS[kind]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, kind, (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(output), canvas)


def error_count(record):
    return sum(len(record['feedback'][key]) for key in FEEDBACK_TYPES
               if key != 'matched')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feedback', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-images', type=int, default=50)
    parser.add_argument('--img-size', type=int, default=640)
    args = parser.parse_args()
    feedback_path = Path(args.feedback).resolve()
    records = [json.loads(line) for line in
               feedback_path.open(encoding='utf-8') if line.strip()]
    paths = [Path(record['image_path']) for record in records]
    if len({record['image_id'] for record in records}) != len(records):
        raise ValueError('Duplicate image_id in feedback')
    if any('train' not in [part.lower() for part in path.parts]
           for path in paths):
        raise ValueError('Feedback contains non-train images')
    report = analyze(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / 'feedback_quality_report.json'
    with open(report_path, 'w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    dataset = PestDataset(args.data_root, 'train', args.img_size,
                          transforms=get_val_transforms(args.img_size))
    selected = sorted(records, key=error_count, reverse=True)
    selected = [record for record in selected if error_count(record) > 0]
    for rank, record in enumerate(selected[:args.num_images]):
        name = '{:03d}_image_{}.jpg'.format(rank, record['image_id'])
        draw_record(record, dataset, output_dir / name)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'Saved report: {report_path}')
    print(f'Saved visualizations: {min(args.num_images, len(selected))}')


if __name__ == '__main__':
    main()
