import argparse
from evaluate import _load_model_for_eval, evaluate_checkpoint
from dataloader import get_pest_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    supervised = '../kb1_reward_guided_training/checkpoints/yolov8n/weights/best.pt'
    loader = get_pest_dataloader('../pre-data/data/v2i', 'val', 16, 640)
    model = _load_model_for_eval(
        args.checkpoint, 'ultralytics', args.device,
        supervised_ckpt=supervised,
    )
    print(evaluate_checkpoint(model, loader, 'ultralytics', args.device), flush=True)


if __name__ == '__main__':
    main()
