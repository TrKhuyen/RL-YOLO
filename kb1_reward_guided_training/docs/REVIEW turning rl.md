# Review chi tiết: tuning_cv_models_with_rl_torch

> **Nguồn gốc**: Repo thực nghiệm dựa trên paper  
> **"Tuning computer vision models with task rewards"** – Google DeepMind, 2023  
> arxiv: https://arxiv.org/abs/2302.08242  
> Tác giả repo: yanivnik  
> Trạng thái: **Work In Progress (WIP)**

---

## Mục lục
1. [Project làm gì?](#1-project-làm-gì)
2. [Kiến trúc tổng thể](#2-kiến-trúc-tổng-thể)
3. [Setup môi trường](#3-setup-môi-trường)
4. [Luồng huấn luyện (Training Flow)](#4-luồng-huấn-luyện)
5. [RL được dùng như thế nào? (Phần cốt lõi)](#5-rl-được-dùng-như-thế-nào)
6. [Chi tiết từng file](#6-chi-tiết-từng-file)
7. [Điểm mạnh / Điểm yếu / TODO còn lại](#7-điểm-mạnh--điểm-yếu--todo-còn-lại)

---

## 1. Project làm gì?

### Ý tưởng chính

Project **fine-tune mô hình Computer Vision (Object Detection) bằng Reinforcement Learning** thay vì supervised loss thông thường.

Thay vì tối ưu cross-entropy / regression loss trên từng bounding box, project dùng **toàn bộ task metric** (ví dụ: Recall, mAP) làm **reward** để huấn luyện mô hình theo hướng policy gradient (REINFORCE).

> **Dẫn chứng – README.md dòng 1–2:**
> ```
> Experimentation repo for the "Tuning computer vision models with task rewards" paper
> This repository includes some of my code experimentations with the ideas presented in the paper
> [Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242).
> ```

### Vì sao cần fine-tune bằng RL?

Supervised loss (cross-entropy, L1 regression…) tối ưu **proxy metric** – không phải metric cuối mô hình được đánh giá. Điều này tạo ra **train-test metric gap**:

- Mô hình có thể giảm loss nhưng mAP/Recall không tăng tương xứng.
- Supervised loss không phân biệt được các lỗi "nặng" hay "nhẹ" theo quan điểm task thực tế.

RLHF (Reinforcement Learning from Human Feedback) và REINFORCE cho phép tối ưu **trực tiếp task metric**, dù metric đó không khả vi.

> Đây chính là ý tưởng của paper gốc, áp dụng tương tự như RLHF trong NLP (InstructGPT) nhưng cho CV.

---

## 2. Kiến trúc tổng thể

```
┌─────────────────────────────────────────────┐
│                   main.py                   │
│  1. evaluate() baseline                     │
│  2. reward_finetune() – RL training loop    │
│  3. evaluate() post fine-tuning             │
│  4. save model checkpoint                   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│          tasks/object_detection.py          │
│  class DetectionTask                        │
│  ├── _build_model()      → DETR ResNet-50   │
│  ├── _build_dataloader() → COCO 2017        │
│  ├── evaluate()          → mAP + Recall     │
│  ├── reward_finetune()   → REINFORCE loop   │
│  ├── compute_reward()    → recall / mAP     │
│  └── detect_objects()    → forward pass     │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴────────┐
          ▼                 ▼
┌──────────────────┐  ┌───────────────────────┐
│ coco_detection_  │  │  detection_utils.py   │
│ dataset.py       │  │  - bbox conversion    │
│ CocoDetection    │  │  - visualization      │
│ (HF DETR wrapper)│  └───────────────────────┘
└──────────────────┘
```

### Mô hình: **DETR (Detection Transformer)**
- Pretrained: `facebook/detr-resnet-50`
- Framework: HuggingFace Transformers + PyTorch
- Backbone: ResNet-50 → Transformer encoder-decoder → FFN → (boxes, labels, scores)

> **Dẫn chứng – object_detection.py dòng 26:**
> ```python
> model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True)
> ```

### Dataset: **COCO 2017**
- Split train/val chuẩn COCO
- Annotation format: `instances_train2017.json`, `instances_val2017.json`

---

## 3. Setup môi trường

### Bước 1 – Clone và cài package

```bash
git clone <repo>
cd tuning_cv_models_with_rl_torch
pip install torch torchvision transformers torchmetrics supervision tensorboard tqdm matplotlib numpy
```

Các package chính:
| Package | Vai trò |
|---|---|
| `torch` | Deep learning framework |
| `transformers` | DETR model + DetrImageProcessor |
| `torchvision` | CocoDetection base dataset, box_iou |
| `torchmetrics` | Tính MeanAveragePrecision |
| `supervision` | Vẽ bounding box lên ảnh |
| `tensorboard` | TensorBoard logging |

### Bước 2 – Cấu hình dataset

```bash
cp settings.py.default settings.py
```

Sau đó sửa `settings.py`:
```python
datasets_dir = '/PATH/TO/COCO/datasets/COCO_2017'
```

Cấu trúc thư mục COCO 2017 cần:
```
COCO_2017/
├── train2017/          ← ảnh train
├── val2017/            ← ảnh val
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

> **Dẫn chứng – settings.py.default:**
> ```python
> datasets_dir = '/PATH/TO/COCO/datasets/COCO_2017'
> ```
> 
> **Dẫn chứng – coco_detection_dataset.py dòng 13–14:**
> ```python
> data_dir = os.path.join(root_dir, f'{split}2017')
> annotation_file_path = os.path.join(root_dir, 'annotations', f'instances_{split}2017.json')
> ```

### Bước 3 – Chạy

```bash
python main.py
```

---

## 4. Luồng huấn luyện

### Pipeline đầy đủ (main.py)

```
1. set_deterministic(seed=42)          ← đảm bảo reproducibility
       │
2. DetectionTask(datasets_dir, batch_size=16)
       │   ├─ load DETR pretrained
       │   ├─ build train DataLoader (COCO train2017)
       │   └─ build val DataLoader   (COCO val2017)
       │
3. detection.evaluate()                ← baseline mAP + Recall@100
       │
4. detection.reward_finetune(steps=100_000)   ← RL fine-tuning
       │
5. detection.evaluate()                ← post fine-tuning metrics
       │
6. torch.save(model.state_dict(), ...)  ← lưu checkpoint
```

### Chi tiết reward_finetune() – Vòng lặp RL

```python
# object_detection.py dòng 57–80
optim = torch.optim.Adam(self.model.parameters(), lr=1e-6)

for step in range(steps):           # 100,000 steps
    batch = next(dataloader_iter)   # lấy batch từ COCO train
    
    # 1. Forward pass (inference) → lấy predictions
    preds = DetectionTask.detect_objects(...)
    
    # 2. Tính reward từ task metric (recall)
    rewards = DetectionTask.compute_reward(preds, targets, type='recall')
    
    # 3. Tính REINFORCE loss
    avg_confidences = torch.stack([torch.mean(p['scores']).nan_to_num() for p in preds])
    loss = -1 * torch.mean(torch.log(avg_confidences) * rewards)
    
    # 4. Backprop và update
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # 5. Log lên TensorBoard
    tb_logger.add_scalar('detection/reward', ...)
    tb_logger.add_scalar('detection/loss',   ...)
```

**Optimizer**: Adam, lr = 1e-6 (rất nhỏ, tránh catastrophic forgetting)  
**Steps**: 100,000  
**Logging**: TensorBoard, lưu vào `./results/tensorboard/<run_id>/`

---

## 5. RL được dùng như thế nào?

> Đây là phần **quan trọng nhất** của project.

### 5.1 Mapping sang bài toán RL

| RL concept | Trong project này |
|---|---|
| **Environment** | Dataset COCO (ảnh đầu vào) |
| **Agent / Policy** | Mô hình DETR |
| **State** | Ảnh đầu vào (`pixel_values`) |
| **Action** | Tập bounding box predictions (boxes, labels, scores) |
| **Reward** | Recall-based score hoặc mAP (task metric) |
| **Log-probability** | `log(avg_confidence_score)` |

### 5.2 Thuật toán: REINFORCE (Policy Gradient)

REINFORCE cập nhật policy bằng gradient:

$$\nabla_\theta J(\theta) = \mathbb{E}_\tau \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot R \right]$$

Trong code:

```python
# object_detection.py dòng 72–73
avg_confidences = torch.stack([
    torch.mean(p['scores']).nan_to_num() for p in preds
]).clamp(1e-20, 1.0)

loss = -1 * torch.mean(torch.log(avg_confidences) * rewards)
#  │                │                                    │
#  └─ gradient      └─ log π(a|s)                        └─ R (reward)
#     ascent           (log-likelihood proxy)
```

- `-1` ở đầu: vì PyTorch tối thiểu hóa loss, nên thêm `-1` để biến gradient descent → gradient ascent (maximize reward).
- `torch.log(avg_confidences)`: confidence score của detection (∈ [0,1]) được dùng làm **xấp xỉ log-likelihood** của policy. Đây là một **simplification** – thay vì tính log-prob đầy đủ của toàn bộ decoder output, ta dùng average confidence.
- `rewards`: scalar reward cho từng ảnh trong batch.

> **Dẫn chứng – object_detection.py dòng 70–75:**
> ```python
> # We treat the average confidence of all BBox predictions in a given image as the likelihood estimate
> avg_confidences = torch.stack([torch.mean(p['scores']).nan_to_num() for p in preds]).clamp(1e-20, 1.0)
> loss = -1 * torch.mean(torch.log(avg_confidences) * rewards)  # -1 because we want to perform gradient ascent
> ```

### 5.3 Hàm Reward: compute_reward()

#### Reward type = 'recall' (được dùng trong training)

```python
# object_detection.py dòng 87–118
for i in range(len(preds)):             # với mỗi ảnh trong batch
    for cls in classes:                 # với mỗi class có trong ground truth
        
        # Tính IoU matrix: GT boxes × Predicted boxes
        iou_matrix = torchvision.ops.box_iou(target_boxes, pred_boxes)
        iou_matrix = iou_matrix > iou_threshold   # threshold = 0.8
        
        # Số GT box được match với ít nhất 1 prediction
        count_of_matched_gt_boxes = torch.any(iou_matrix, dim=-1).sum()
        
        # Số prediction trùng lặp (match cùng 1 GT box)
        count_of_duplicate_boxes = (torch.sum(iou_matrix, dim=-1) - 1).clamp(0).sum()
        
        # Reward = matched - 0.3 * duplicate
        recall_rewards[i] += (count_of_matched_gt_boxes - count_of_duplicate_boxes * 0.3)
    
    recall_rewards[i] /= len(classes)   # average over classes
```

**Giải thích logic reward**:
- `count_of_matched_gt_boxes`: khuyến khích mô hình **tìm được nhiều object** (tăng recall).
- `- count_of_duplicate_boxes * 0.3`: **phạt nhẹ** khi mô hình predict nhiều box cho cùng 1 object (duplicate). Hệ số 0.3 nhỏ hơn 1.0 có chủ đích – không phạt quá nặng vì duplicate ít ảnh hưởng hơn là miss object.
- `iou_threshold = 0.8`: khá nghiêm ngặt, chỉ tính là "match" khi IoU ≥ 0.8.
- `/ len(classes)`: normalize reward, không để ảnh có nhiều class nhận reward cao hơn không công bằng.

#### Reward type = 'map' (thực nghiệm, có bugs - chưa hoàn thiện)

```python
# object_detection.py dòng 120–124
mean_aps_sep = [MeanAveragePrecision()([preds[i]], [targets[i]])['map'].detach() 
                for i in range(len(preds))]
return torch.stack(mean_aps_sep)
```

> **Dẫn chứng – comment trong code:**
> ```python
> # This is a different implementation from the paper, for a per-example mAP reward.
> # It currently has some problems, as I didn't fully understand which supervised loss did the authors use
> # in this section and how did it combine with the recall reward.
> ```

### 5.4 Tại sao dùng RL thay vì supervised fine-tuning?

**Vấn đề của supervised fine-tuning**:
1. Loss function (cross-entropy, Hungarian matching loss trong DETR) không tương đương trực tiếp với mAP hay Recall.
2. DETR dùng **bipartite matching loss** trong training, nhưng metric thực tế là mAP – hai thứ này không hoàn toàn tương quan.
3. Không thể backprop qua **non-differentiable evaluation metrics** (IoU threshold, Recall computation).

**RL giải quyết bằng cách**:
1. Dùng reward = task metric (Recall), không cần metric phải khả vi.
2. Gradient truyền qua `log(confidence)` – phần khả vi của mô hình.
3. Cho phép mô hình trực tiếp tối ưu metric sẽ được đánh giá.

> Đây là cách tiếp cận tương tự RLHF trong NLP:
> - NLP: reward = human preference score, optimize language model policy
> - CV: reward = recall/mAP score, optimize detection model policy

### 5.5 Sự đơn giản hóa (Simplifications) so với REINFORCE đầy đủ

| Đặc điểm đầy đủ | Trong code này |
|---|---|
| Log-prob = log P(toàn bộ output sequence) | Chỉ dùng `log(avg_confidence)` |
| Baseline (giảm variance) | Chưa implement (TODO trong code) |
| Episode = nhiều steps | 1 batch = 1 "episode" |
| Discount factor γ | Không có (single-step reward) |

> **Dẫn chứng – TODO trong code (object_detection.py dòng 65):**
> ```python
> # baseline = ... # TODO UNDERSTAND WHAT SHOULD BE USED AS THE BASELINE, IF ANY
> ```

---

## 6. Chi tiết từng file

### main.py
Entry point. Thực hiện tuần tự:
1. `set_deterministic()` – seed 42 cho reproducibility
2. Tạo `DetectionTask` (load model + dataloader)
3. `evaluate()` – in baseline metrics
4. `reward_finetune(steps=100_000)` – RL training
5. `evaluate()` – in post-training metrics
6. `torch.save()` – lưu model

**Lưu ý**: Có comment hardcode baseline metrics để skip evaluate() cho nhanh (dòng 13–14):
```python
# original_mAP, original_recall = torch.tensor(0.3959), torch.tensor(0.4857) 
# Found after total evaluation, hardcoded for now to save time.
```

### settings.py.default
File template cấu hình. Chỉ 1 biến:
```python
datasets_dir = '/PATH/TO/COCO/datasets/COCO_2017'
```
Cần copy thành `settings.py` và sửa path.

### utils.py
Ba utility function:
- `set_deterministic(seed=42)`: fix seed cho torch, numpy, random, cudnn.
- `get_tb_logger()`: tạo TensorBoard SummaryWriter với run ID ngẫu nhiên dạng `xayu-Apr06_12-00-00`.
- `to_device(obj, device)`: đệ quy chuyển Tensor/list/dict sang device.

### tasks/coco_detection_dataset.py
Subclass của `torchvision.datasets.CocoDetection`, adapt cho DETR:
- Dùng `DetrImageProcessor` để preprocess ảnh + annotation.
- `__getitem__`: trả về `(pixel_values, target, orig_size)`.
- `coco_collate_fn`: padding batch về cùng size (cần thiết vì ảnh COCO khác kích thước).
- Chuyển bbox từ relative `[0,1]` → absolute `[0,H/W]` pixels.

> **Dẫn chứng – coco_detection_dataset.py dòng 1:**
> ```python
> # Code based on: https://github.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb
> ```

### tasks/detection_utils.py
- `get_detection_image()`: vẽ bounding box lên ảnh dùng `supervision` library.
- `relative_to_absolute_bboxes()`: chuyển `center_x, center_y, w, h` (relative) → `x1, y1, x2, y2` (absolute pixels).

### tasks/object_detection.py
File core. Class `DetectionTask`:
- `_build_model()`: load `facebook/detr-resnet-50` pretrained, đưa lên CUDA.
- `_build_dataloader()`: tạo DataLoader với custom `coco_collate_fn`.
- `evaluate()`: vòng lặp val set, tính `MeanAveragePrecision` từ torchmetrics, trả về `(map, mar_100)`.
- `reward_finetune()`: **vòng lặp RL chính** (xem phần 5).
- `compute_reward()`: tính reward recall hoặc mAP per image.
- `detect_objects()`: `@torch.no_grad()` là decorator → inference không tính gradient cho predictions, sau đó post-process output của DETR.

---

## 7. Điểm mạnh / Điểm yếu / TODO còn lại

### Điểm mạnh
- Ý tưởng sạch, trực tiếp map paper sang code.
- Dùng HuggingFace pretrained DETR – không cần train từ đầu.
- Reward function có tư duy tốt: phạt duplicate nhẹ (0.3×), normalize theo số class.
- TensorBoard logging có sẵn.
- `set_deterministic` đảm bảo reproducibility.

### Điểm yếu / Hạn chế kỹ thuật

1. **Log-prob approximation thô**: `log(avg_confidence)` là proxy rất đơn giản cho log-likelihood của policy. DETR decoder là một sequence model, log-prob thực sự phức tạp hơn nhiều.

2. **Không có baseline**: REINFORCE có variance cao. Thông thường cần trừ baseline (ví dụ: exponential moving average của reward) để ổn định training.
   ```python
   # object_detection.py dòng 65
   # baseline = ... # TODO UNDERSTAND WHAT SHOULD BE USED AS THE BASELINE, IF ANY
   ```

3. **mAP reward chưa hoàn thiện**: Phần mAP reward có bug, tác giả tự nhận chưa hiểu rõ paper gốc kết hợp supervised loss + reward thế nào.

4. **Không có CLI**: mọi tham số hardcode, không có argparse.
   ```python
   # main.py dòng 8
   # TODO: ADD COMMAND LINE ARGUMENTS FOR TASK SELECTION, PARAMETER SETTING, ETC.
   ```

5. **Chỉ implement detection**: Paper gốc cover nhiều task CV khác (segmentation, captioning...).
   ```
   # README: Need to implement the logic for more tasks.
   ```

6. **Baseline mAP hardcode** (commented): Tác giả dùng hardcoded baseline để tiết kiệm thời gian debug, chưa phải flow production chuẩn.

### TODO còn lại (theo README + code comments)

| TODO | Vị trí | Ưu tiên |
|---|---|---|
| Debug recall fine-tuning matching paper trend | README | Cao |
| Implement baseline cho REINFORCE | object_detection.py:65 | Cao |
| Fix mAP reward implementation | object_detection.py:120 | Trung bình |
| Add CLI arguments | main.py:8 | Thấp |
| Implement more tasks (segmentation...) | README | Thấp |

---

## Tóm tắt ngắn gọn

```
Paper:   "Tuning CV models with task rewards" (Google DeepMind, 2023)
Goal:    Fine-tune DETR object detector bằng RL thay vì supervised loss
RL alg:  REINFORCE (policy gradient)
Policy:  DETR (facebook/detr-resnet-50)
Reward:  Recall-based (matched GT boxes - 0.3 × duplicate boxes)
Log-π:   log(average_confidence_score)  ← simplification
Dataset: COCO 2017
Status:  WIP, recall fine-tuning chưa match kết quả paper
```
