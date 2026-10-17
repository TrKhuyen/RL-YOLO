"""Resolve KB1's actual training recipe and its original pretrained source."""

from pathlib import Path

from ..run_all import ROOT, MODELS
from .detector import file_hash
from .reference import read_recipe, prepare_reference


# Include native validation settings: they control native best/patience selection.
TRAIN_RECIPE_KEYS = (
    'optimizer', 'nbs', 'amp', 'cos_lr', 'close_mosaic', 'warmup_momentum', 'warmup_bias_lr',
    'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective',
    'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'cutmix', 'copy_paste', 'copy_paste_mode',
    'box', 'cls', 'dfl', 'pose', 'kobj', 'rect', 'single_cls', 'fraction', 'multi_scale',
    'deterministic', 'freeze', 'dropout', 'cache', 'overlap_mask', 'mask_ratio',
    'val', 'conf', 'iou', 'max_det', 'agnostic_nms', 'classes', 'half', 'save_json',
)
INHERITED_SETTINGS = {
    'epochs': 'epochs', 'patience': 'patience', 'seed': 'seed', 'batch_size': 'batch',
    'img_size': 'imgsz', 'workers': 'workers', 'device': 'device',
    'warmup_epochs': 'warmup_epochs', 'lrf': 'lrf',
}


def resolve_pretrained_recipe(model):
    raw, _, path = read_recipe(model)
    if raw.get('pretrained') is not True or raw.get('resume'):
        raise ValueError('KB1 YAML must specify a fresh pretrained training session')
    if raw.get('time') is not None or raw.get('val') is not True:
        raise ValueError('KB1 recipe must use epoch/patience budgeting and validation')
    source = Path(str(raw['model']).replace('\\', '/'))
    if not source.is_absolute():
        source = ROOT / source
    source = source.resolve()
    architecture = Path(MODELS[model]).stem
    if source.suffix.lower() != '.pt' or source.stem != architecture or not source.is_file():
        raise ValueError(f'Original pretrained checkpoint from KB1 YAML is missing or not {architecture}.pt: {source}')
    reference = path.parent / 'weights' / 'best.pt'
    source_sha = file_hash(source)
    if source == reference.resolve() or (reference.is_file() and source_sha == file_hash(reference)):
        raise ValueError('KB1 best.pt is comparison-only; it cannot initialize a trial')
    recipe = {name: raw[name] for name in TRAIN_RECIPE_KEYS if name in raw}
    return raw, recipe, dict(checkpoint=str(source), checkpoint_sha256=source_sha,
                            role='original_pretrained_source_from_kb1_yaml',
                            architecture=architecture, recipe_path=str(path.resolve()),
                            recipe_sha256=file_hash(path))


def verify_pretrained_source(source):
    from ultralytics import YOLO
    checkpoint = Path(source['checkpoint'])
    if file_hash(checkpoint) != source['checkpoint_sha256']:
        raise RuntimeError('Original pretrained source checkpoint changed')
    detector = YOLO(str(checkpoint))
    if not detector.ckpt or detector.model.yaml.get('nc') != 80:
        raise ValueError('The original KB1 source must be a pretrained 80-class COCO detector')
    # Older official YOLOv8 weights have no yaml_file/scale metadata. Compare
    # the full state tensor layout against the requested architecture instead.
    import torch
    from ultralytics.nn.tasks import DetectionModel
    with torch.random.fork_rng(devices=[]):
        expected = DetectionModel(source['architecture'] + '.yaml', nc=80, verbose=False)
    signature = lambda model: {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}
    if signature(detector.model) != signature(expected):
        raise ValueError('Original pretrained source architecture does not match the selected KB1 model')
    if file_hash(checkpoint) != source['checkpoint_sha256']:
        raise RuntimeError('Pretrained source changed during preflight')
    print(f'PASS original pretrained source: {checkpoint.name}, nc=80, SHA={source["checkpoint_sha256"]}', flush=True)


def prepare_pretrained_reference(args, settings, space):
    # Reuse the verified, read-only KB1 baseline and actual initial hyperparameters.
    space = prepare_reference(args, settings, space)
    raw, recipe, source = resolve_pretrained_recipe(args.model)
    if source != settings['pretrained_source']:
        raise ValueError('KB1 pretrained source/recipe changed during preflight')
    if not args.smoke:
        import torch
        payload = torch.load(settings['kb1_reference']['checkpoint'], map_location='cpu', weights_only=False)
        for name, value in recipe.items():
            if payload['train_args'].get(name) != value:
                raise ValueError(f'KB1 YAML differs from its supervised checkpoint: {name}')
    else:
        recipe = {**recipe, 'nbs': settings['batch_size'], 'close_mosaic': 1}
    settings['trainer_recipe'] = recipe
    verify_pretrained_source(source)
    return space
