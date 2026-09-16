import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from dataloader import PestDataset, get_val_transforms, pest_collate_fn
from feedback import FEEDBACK_TYPES, SCHEMA_VERSION, build_feedback_record
from train_rl import CHECKPOINTS, REPO_ROOT, SCRIPT_DIR, load_adapter


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_loader(data_root, batch_size, img_size, limit):
    dataset = PestDataset(data_root, 'train', img_size,
                          transforms=get_val_transforms(img_size))
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, collate_fn=pest_collate_fn)


def generate(args):
    checkpoint = Path(args.checkpoint or CHECKPOINTS[args.model]).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output or
                  SCRIPT_DIR / 'feedback_data' / f'{args.model}_train.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    loader = make_loader(args.data_root, args.batch_size,
                         args.img_size, args.limit)
    adapter = load_adapter(args.model, str(checkpoint), args.device)
    adapter.eval_mode()
    counts, images_with = Counter(), Counter()
    num_images = num_preferences = 0

    with open(temporary, 'w', encoding='utf-8') as stream, torch.no_grad():
        for batch_i, (images, targets) in enumerate(loader, 1):
            predictions = adapter.forward_with_grad(
                images.to(args.device), args.conf, args.nms_iou)
            for prediction, target in zip(predictions, targets):
                record = build_feedback_record(
                    prediction, target, args.match_iou,
                    args.localization_iou)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                num_images += 1
                num_preferences += len(record['preferences'])
                for kind in FEEDBACK_TYPES:
                    n = len(record['feedback'][kind])
                    counts[kind] += n
                    images_with[kind] += int(n > 0)
            print(f'feedback {batch_i}/{len(loader)} batches', flush=True)
    temporary.replace(output)
    summary = {
        'schema_version': SCHEMA_VERSION,
        'split': 'train',
        'model': args.model,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': sha256(checkpoint),
        'data_root': str(Path(args.data_root).resolve()),
        'img_size': args.img_size,
        'conf_threshold': args.conf,
        'nms_iou_threshold': args.nms_iou,
        'match_iou_threshold': args.match_iou,
        'localization_iou_threshold': args.localization_iou,
        'num_images': num_images,
        'feedback_counts': dict(counts),
        'images_with_feedback': dict(images_with),
        'num_preferences': num_preferences,
    }
    summary_path = output.with_suffix('.summary.json')
    with open(summary_path, 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved: {output}')
    print(f'Saved: {summary_path}')
    return output, summary_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate train-only feedback for YOLO fine-tuning')
    parser.add_argument('--model', default='yolov8n', choices=CHECKPOINTS)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--data-root',
                        default=str(REPO_ROOT / 'pre-data/data/v2i'))
    parser.add_argument('--output', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--nms-iou', type=float, default=0.45)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--localization-iou', type=float, default=0.1)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error('--limit must be positive')
    generate(args)


if __name__ == '__main__':
    main()
