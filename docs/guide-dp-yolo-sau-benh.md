# HƯỚNG DẪN XÂY DỰNG DP-YOLO CHO BÀI TOÁN PHÁT HIỆN SÂU BỆNH

> **Tổng hợp từ 3 bài báo:**
> - DP-YOLO gốc (Wang et al., 2023 – Applied Sciences/MDPI)
> - DP-YOLO cho phát hiện sâu bệnh (kích thước nhỏ, thiết bị nhúng)
> - DP-YOLO cho phát hiện lỗi chốt kẹp ray (Sensors/MDPI, 2025)

---

## 1. ĐẶT VẤN ĐỀ VÀ LỰA CHỌN BASELINE

### 1.1. Tại sao chọn YOLOv5s làm baseline?
Cả 3 bài báo đều lấy **YOLOv5s** làm điểm khởi đầu vì:
- Cân bằng tốt giữa độ chính xác và tốc độ (75 FPS trên RTX 3090).
- Kiến trúc nhẹ (7.2M tham số, 16.4 GFLOPS), phù hợp triển khai trên thiết bị nhúng ngoài đồng.
- Cộng đồng lớn, dễ huấn luyện và fine-tune.
- Quan trọng nhất: toàn bộ logic cải tiến của DP-YOLO trong 3 bài đều được thiết kế như **một nhánh nâng cấp trực tiếp từ YOLOv5s**, nên chọn YOLOv5s giúp tái hiện thí nghiệm và kiểm soát ablation rõ ràng hơn.

### 1.2. Vị trí của YOLOv8 và YOLOv11 trong bối cảnh hiện nay
Trong bối cảnh triển khai thực tế năm 2026, **YOLOv8** và **YOLOv11** là các baseline mới đáng cân nhắc vì có nhiều cải tiến về head, chiến lược huấn luyện và hiệu quả suy luận. Tuy nhiên, chúng không thay thế trực tiếp vai trò của YOLOv5s trong guide này.

| Mô hình | Điểm mạnh | Hạn chế khi dùng làm nền để xây DP-YOLO |
|---|---|---|
| YOLOv5s | Ổn định, nhẹ, dễ sửa kiến trúc, bám sát 3 bài nguồn | Label assignment và phát hiện vật thể nhỏ chưa tối ưu |
| YOLOv8n/s | Anchor-free, pipeline hiện đại hơn, tổng quát tốt hơn trên nhiều benchmark | Khác nền kiến trúc với các bài DP-YOLO gốc, khó giữ so sánh ablation công bằng |
| YOLOv11n/s | Tối ưu tốt hơn cho hiệu năng và triển khai, là lựa chọn mạnh cho baseline mới | Nếu chuyển sang YOLOv11 thì guide không còn là tái dựng DP-YOLO theo bài báo mà trở thành một nhánh thiết kế mới |

**Kết luận lựa chọn baseline:**
- Nếu mục tiêu là **tái hiện và phát triển tiếp đúng tinh thần DP-YOLO trong tài liệu nguồn**, nên bắt đầu từ **YOLOv5s**.
- Nếu mục tiêu là **đạt trần hiệu năng mới trong sản phẩm**, có thể dùng YOLOv8 hoặc YOLOv11 làm baseline đối chứng bổ sung ở giai đoạn sau.
- Cách làm chặt nhất là: xây bản DP-YOLO trên YOLOv5s trước, sau đó mới chuyển từng cải tiến sang YOLOv8/YOLOv11 để đo mức đóng góp thực sự.

### 1.3. Những hạn chế của YOLOv5s với bài toán sâu bệnh
| Vấn đề | Biểu hiện |
|---|---|
| Phát hiện vật thể nhỏ kém | Sâu, trứng, đốm bệnh kích thước nhỏ bị bỏ sót |
| Chiến lược gán nhãn tĩnh | Bỏ lỡ mẫu dương tính khi vật thể ở vị trí biên |
| Thiếu khả năng xử lý biến dạng | Sâu uốn cong, lá bị cuộn, tư thế thay đổi liên tục |
| Nền phức tạp | Lá cây, đất, ánh sáng thay đổi gây nhầm lẫn |

---

## 2. KIẾN TRÚC DP-YOLO ĐỀ XUẤT CHO SÂU BỆNH

Kết hợp có chọn lọc từ cả 3 bài, kiến trúc tối ưu gồm **5 cải tiến trụ cột**:

```
INPUT IMAGE (640×640)
        │
┌───────▼────────────────────────────────┐
│  BACKBONE: DYB + DBBNCSPELAN4          │
│  (Deformable Conv + DBB Reparameter.)  │
└───────┬────────────────────────────────┘
        │
┌───────▼────────────────────────────────┐
│  NECK: PTCSP + C3Ghost                 │
│  (Transformer + Ghost Feature Fusion)  │
└───────┬────────────────────────────────┘
        │
┌───────▼────────────────────────────────┐
│  DETECTION HEAD: P2 + P3 + P4          │
│  (Bỏ P5, thêm P2 cho vật thể cực nhỏ) │
└───────┬────────────────────────────────┘
        │
┌───────▼────────────────────────────────┐
│  LABEL ASSIGNMENT: PSA                 │
│  (Petal-like Sample Amplification)     │
└───────┬────────────────────────────────┘
        │
┌───────▼────────────────────────────────┐
│  LOSS: W3F_MPDIoU                      │
└────────────────────────────────────────┘
```

---

## 3. CÁC CẢI TIẾN CHI TIẾT

### 3.1. Thay đổi Detection Head: Loại P5, Thêm P2

**Lý do:**
Sâu bệnh trên lá, thân cây thường có kích thước rất nhỏ so với toàn bộ khung ảnh. Lớp phát hiện P5 (stride 32) được thiết kế cho vật thể lớn, không cần thiết trong bài toán này.

**Cấu hình mới:**

| Head | Stride | Feature Map (640px input) | Mục tiêu |
|------|--------|--------------------------|----------|
| P2   | 4      | 160 × 160                | Sâu cực nhỏ, trứng, đốm bệnh |
| P3   | 8      | 80 × 80                  | Sâu nhỏ, vết bệnh nhỏ |
| P4   | 16     | 40 × 40                  | Vùng bệnh trung bình |
| ~~P5~~ | ~~32~~ | ~~20 × 20~~            | **Bỏ** |

**Cách triển khai trong `model.yaml`:**
```yaml
# Thay thế head configuration
head:
  - [-1, 1, Conv, [256, 1, 1]]
  - [[-1, 6], 1, Concat, [1]]   # P4
  - [-1, 3, C3, [256, False]]
  - [-1, 1, Conv, [128, 1, 1]]
  - [[-1, 4], 1, Concat, [1]]   # P3
  - [-1, 3, C3, [128, False]]
  - [-1, 1, Conv, [64, 1, 1]]
  - [[-1, 2], 1, Concat, [1]]   # P2 (MỚI)
  - [-1, 3, C3, [64, False]]
  # 3 detect layers: P2, P3, P4 (không có P5)
  - [[small_idx, medium_idx, large_idx], 1, Detect, [nc, anchors]]
```

---

### 3.2. Backbone: DYB (Deformable YOLO v5 Backbone)

**Nguyên lý:** Thay tích chập thông thường trong Bottleneck của C3 bằng **Deformable Convolution**, cho phép mạng tự học offset để lấy mẫu tại các vị trí linh hoạt – xử lý tốt hơn khi sâu uốn khúc, lá bị biến dạng.

**Hai loại module:**

| Module | Operator | Phù hợp | Tham số |
|--------|----------|---------|---------|
| **D2C3** | DCNv2 | Giai đoạn đầu (kênh ít) | Nhiều hơn (có modulation weight) |
| **D3C3** | DCNv3 | Giai đoạn cuối (kênh nhiều) | Nhẹ hơn (group conv) |

**Chiến lược cấu hình tối ưu (3+1):**
- Stage 1, 2, 3 → dùng **D2C3**
- Stage 4 (downsampling cuối) → dùng **D3C3**

**Lý do chọn chiến lược 3+1:**
> Các tầng đầu cần trích xuất đặc trưng chi tiết (texture, màu sắc), DCNv2 với modulation weight mạnh hơn. Tầng cuối có kênh nhiều, DCNv3 gộp nhóm giữ nhẹ mô hình.

**Code module D2C3 (PyTorch):**
```python
import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.mask_conv = nn.Conv2d(
            in_channels, kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return deform_conv2d(x, offset, self.weight, mask=mask, padding=1)

class D2C3(nn.Module):
    """DCNv2-based C3 bottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1)
        self.cv3 = nn.Conv2d(2 * c_, c2, 1)
        self.m = nn.Sequential(*[
            DeformableConv2d(c_, c_) for _ in range(n)
        ])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
```

---

### 3.3. Backbone: DBBNCSPELAN4 (Diverse Branch Block)

**Nguyên lý:** Trong lúc huấn luyện, dùng **4 nhánh tích chập song song** để học đặc trưng đa dạng. Khi inference, hợp nhất (reparameterize) tất cả thành **1 tích chập 3×3** duy nhất → không tăng tải tính toán khi triển khai thực tế.

**4 nhánh trong training:**
1. Tích chập `1×1`
2. Tích chập `3×3`
3. Average Pooling
4. Identity (nếu in_channels == out_channels)

**Code DBB đơn giản:**
```python
class DiverseBranchBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
        self.avgpool = nn.AvgPool2d(3, stride, 1) if stride > 1 else nn.Identity()
        self.bn3x3 = nn.BatchNorm2d(out_channels)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        self.bn_avg = nn.BatchNorm2d(out_channels)
        self.deployed = False

    def forward(self, x):
        if self.deployed:
            return self.reparam_conv(x)
        out = self.bn3x3(self.conv3x3(x))
        out += self.bn1x1(self.conv1x1(x))
        # Thêm avg pooling branch khi stride=1
        return out

    def reparameterize(self):
        """Gộp tất cả nhánh vào 1 conv 3x3 khi chuyển sang inference"""
        # ... (implementation chi tiết theo công thức từ paper DBB gốc)
        self.deployed = True
```

> **Lưu ý thực tế:** Nên dùng thư viện `timm` hoặc tham khảo repo DiverseBranchBlock gốc để có implementation đầy đủ phần reparameterization.

---

### 3.4. Neck: PTCSP (Partially Transformer CSP)

**Nguyên lý:** Chia kênh đầu vào thành 2 luồng để tận dụng cả CNN lẫn Transformer:
- **Luồng CNN:** Trích xuất đặc trưng cục bộ (texture, edge) → quan trọng cho nhận dạng màu sắc bệnh.
- **Luồng Transformer (MHSA + CGLU):** Nắm bắt ngữ cảnh toàn cục → phân biệt vùng bệnh với nền lá phức tạp.

```
Input Feature Map
        │
   ┌────┴────┐
   │ Split   │  (chia đôi số kênh)
   ▼         ▼
 CNN Path  Transformer Path
 (C3/CSP)  (MHSA + CGLU)
   │         │
   └────┬────┘
        │ Concat
        ▼
  Output Feature Map
```

**Code PTCSP rút gọn:**
```python
class MHSA_CGLU(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cglu = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # Self-attention
        x_flat = x.flatten(2).transpose(1, 2)  # B, HW, C
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x = x + attn_out.transpose(1, 2).view(B, C, H, W)
        # CGLU
        x = x + self.cglu(x)
        return x

class PTCSP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cnn_path = C3(c1 // 2, self.c, n=1)
        self.transformer_path = MHSA_CGLU(c1 // 2)
        self.proj = nn.Conv2d(self.c * 2, c2, 1)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return self.proj(torch.cat([self.cnn_path(x1),
                                     self.transformer_path(x2)], dim=1))
```

---

### 3.5. Neck: C3Ghost (Giảm tham số)

**Nguyên lý:** Module Ghost chia feature map thành:
- **Đường chính:** Tích chập thông thường trích xuất `n/2` đặc trưng.
- **Đường Ghost:** Dùng depthwise convolution rẻ tiền tạo thêm `n/2` đặc trưng phụ (ghost features).

**Ưu điểm:** Giảm ~50% FLOPs so với C3 thông thường trong phần Neck.

```python
class GhostBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            # Primary conv
            nn.Conv2d(c1, c_, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            # Cheap ghost operation
            nn.Conv2d(c_, c_, k, s, k//2, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
        )
        self.shortcut = nn.Identity() if c1 == c2 else nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        x1 = self.conv(x)
        # Ghost: cheap linear operation
        x2 = nn.functional.conv2d(x1, ...) # depthwise
        return torch.cat([x1, x2], dim=1) + self.shortcut(x)
```

---

### 3.6. Chiến lược Gán Nhãn: PSA (Petal-like Sample Amplification)

**Vấn đề với YOLOv5 gốc:**
YOLOv5 gán nhãn dựa trên ngưỡng tỷ lệ cố định [0.25, 4] với lưới lân cận, dễ bỏ sót ground truth nằm gần biên ô lưới.

**Cải tiến PSA:**
Vẽ đường tròn tâm là **trọng tâm từng ô lưới**, bán kính $r$. Nếu tâm ground truth nằm trong đường tròn, các anchors của ô đó được xem là **mẫu dương tính**.

$$\text{Positive} \Leftrightarrow \sqrt{(\Delta x)^2 + (\Delta y)^2} \leq r$$

**Kết quả:**
- Vùng mở rộng của các ô lân cận chồng lên nhau tạo hình **cánh hoa** → tên PSA.
- Tăng ~5% số mẫu dương tính (từ 10.67M lên 11.2M trên COCO2017).
- Tốt nhất khi $r = 1$ (tức bán kính bằng 1 ô lưới).

**Tác động với sâu bệnh:**
Sâu và vết bệnh thường nhỏ, nằm gần biên ô lưới. PSA giúp tăng đáng kể mẫu dương tính cho các đối tượng nhỏ này.

---

### 3.7. Hàm Mất Mát: W3F_MPDIoU

Kết hợp 3 thành phần:

#### a) MPDIoU – Hồi quy bounding box chính xác hơn

Thay vì chỉ dùng diện tích giao, MPDIoU còn tính khoảng cách hai góc:

$$d_1^2 = (x_1 - x_1^{gt})^2 + (y_1 - y_1^{gt})^2 \quad \text{(góc trên trái)}$$

$$d_2^2 = (x_2 - x_2^{gt})^2 + (y_2 - y_2^{gt})^2 \quad \text{(góc dưới phải)}$$

$$MPDIoU = IoU - \frac{d_1^2 + d_2^2}{w^2 + h^2}$$

$$L_{MPDIoU} = 1 - MPDIoU$$

#### b) Focaler-IoU – Tập trung vào các mẫu khó

$$IoU^{focaler} = \begin{cases} \frac{IoU - d}{u - d}, & \text{nếu } d \leq IoU < u \\ 1, & \text{nếu } IoU \geq u \end{cases}$$

$$L_{F\_MPDIoU} = L_{MPDIoU} + IoU - IoU^{focaler}$$

#### c) WIoU v3 – Trọng số mẫu thích nghi

$$L_{W3F\_MPDIoU} = r \cdot R_{WIoU} \cdot L_{F\_MPDIoU}$$

Trong đó:
$$r = \frac{\beta}{\delta\alpha^{\beta} - \delta}, \qquad R_{WIoU} = \exp\!\left(\frac{(x - x_{gt})^2 + (y - y_{gt})^2}{W_g^2 + H_g^2}\right)$$

**Lý do dùng cho sâu bệnh:** Bộ dữ liệu thực địa thường có nhiều mẫu dễ (lá khỏe mạnh), ít mẫu khó (sâu khuất, bệnh giai đoạn đầu). WIoU tự động giảm trọng số mẫu dễ, tập trung tối ưu vào mẫu khó.

---

## 4. THU THẬP VÀ CHUẨN BỊ DỮ LIỆU

### 4.1. Gợi ý thu thập ảnh
- **Điều kiện đa dạng:** Ánh sáng buổi sáng / chiều / trời mây, độ ẩm khác nhau.
- **Góc chụp:** Thẳng đứng, xiên 45°, sát lá.
- **Thiết bị:** Smartphone (≥12MP), camera máy bay không người lái (UAV) cho diện tích lớn.
- **Tỷ lệ train:val:test = 7:2:1** (tham khảo Urised11: 4:1, Fastener: ~7:3).

### 4.2. Tăng cường dữ liệu (Data Augmentation)
Tham khảo chiến lược từ bài Rail Fastener (mỗi ảnh chọn ngẫu nhiên 3 phương pháp):

```python
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),    # Biến thiên ánh sáng ngoài đồng
    A.HueSaturationValue(p=0.4),           # Màu sắc lá khác nhau theo mùa
    A.GaussNoise(p=0.3),                   # Nhiễu camera
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.4),
    A.RandomCrop(height=512, width=512, p=0.3),
    A.Cutout(num_holes=4, max_h_size=30, max_w_size=30, p=0.3),
    A.CLAHE(p=0.2),                        # Tăng tương phản vùng tối
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### 4.3. Định dạng nhãn (YOLO format)
```
# Mỗi dòng: class_id cx cy w h (tọa độ chuẩn hóa về [0,1])
0 0.512 0.334 0.045 0.038   # sâu đục thân
1 0.231 0.671 0.120 0.095   # vùng bệnh đốm nâu
```

---

## 5. CẤU HÌNH MÔI TRƯỜNG VÀ HUẤN LUYỆN

### 5.1. Môi trường (theo DP-YOLO gốc)
```
OS:      Ubuntu 20.04
Python:  3.9
PyTorch: 1.12.1
CUDA:    11.2
GPU:     NVIDIA RTX 3090 (hoặc 3080/A100)
```

**Cài đặt:**
```bash
conda create -n dp-yolo python=3.9 -y
conda activate dp-yolo
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
pip install torchvision timm albumentations
```

### 5.2. Tham số huấn luyện
```yaml
# hyp.pest.yaml - Hyperparameters cho sâu bệnh
lr0: 0.01          # Learning rate khởi đầu
lrf: 0.01          # Learning rate cuối = lr0 * lrf
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
box: 0.05          # Box regression loss gain
cls: 0.5           # Class loss gain (tăng nếu nhiều lớp)
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
anchor_t: 4.0
```

```bash
python train.py \
  --img 640 \
  --batch 32 \
  --epochs 300 \
  --data pest.yaml \
  --cfg models/dp-yolo-pest.yaml \
  --weights yolov5s.pt \
  --hyp data/hyps/hyp.pest.yaml \
  --optimizer SGD \
  --name dp-yolo-pest
```

### 5.3. File cấu hình dataset `pest.yaml`
```yaml
path: ../datasets/pest
train: images/train
val: images/val
test: images/test

nc: 10  # Số lớp sâu bệnh
names:
  0: sau_duc_than
  1: sau_cuon_la
  2: benh_dom_nau
  3: benh_dao_on
  4: benh_kho_van
  5: ran_xanh
  6: bo_tri
  7: ran_nau
  8: dom_la
  9: healthy
```

---

## 6. ĐÁNH GIÁ MÔ HÌNH

### 6.1. Các chỉ số quan trọng

$$Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{TP + FN}$$

$$AP_i = \int_0^1 P(R)\,dR \quad \text{(cho lớp thứ } i\text{)}$$

$$mAP = \frac{1}{n}\sum_{i=1}^n AP_i$$

- **mAP@0.5:** Đánh giá chính (IoU threshold = 0.5).
- **mAP@0.5:0.95:** Đánh giá nghiêm ngặt hơn, quan trọng cho deployment.
- **$AP_s$:** AP cho vật thể nhỏ (diện tích < 32² px) – chỉ số then chốt với sâu non, trứng.

### 6.2. Ngưỡng hiệu suất tham khảo
| Mô hình | mAP@0.5 | Tham số | FPS |
|---------|---------|---------|-----|
| YOLOv5s (baseline) | ~85-88% | 7.2M | 75 |
| **DP-YOLO (mục tiêu)** | **≥89%** | **≤8M** | **≥60** |

*(Dựa trên kết quả Rail Fastener: tăng 1.3% mAP@0.5, giảm 15% FLOPs)*

---

## 7. KẾT QUẢ KỲ VỌNG VÀ PHÂN TÍCH

### 7.1. Lợi ích từ từng cải tiến

| Cải tiến | Vấn đề giải quyết | Tác động dự kiến |
|----------|------------------|-----------------|
| P2 Detection Head | Phát hiện sâu nhỏ, trứng | $AP_s$ tăng ≥2 AP |
| DYB (Deformable Conv) | Sâu uốn cong, biến dạng | AP các lớp khó tăng 4-8 AP |
| PSA | Bỏ sót mẫu biên ô lưới | Tăng ~5% mẫu dương tính |
| PTCSP | Nền lá phức tạp | Giảm False Positive |
| C3Ghost | Mô hình nặng | Giảm ~50% FLOPs Neck |
| W3F_MPDIoU | Mất cân bằng dữ liệu | Hội tụ nhanh hơn, mAP tổng tăng |

### 7.2. Ablation Study đề xuất
Nên thực hiện ablation theo thứ tự từng thành phần:

```
Exp 1: YOLOv5s baseline
Exp 2: + P2 head (bỏ P5)
Exp 3: + DYB backbone
Exp 4: + PSA label assignment
Exp 5: + PTCSP neck
Exp 6: + C3Ghost neck
Exp 7: + W3F_MPDIoU loss  ← DP-YOLO đầy đủ
```

---

## 8. TRIỂN KHAI THỰC TẾ (DEPLOYMENT)

### 8.1. Export mô hình
```python
# Export sang ONNX (dùng cho Jetson Nano, Raspberry Pi)
python export.py --weights runs/train/dp-yolo-pest/weights/best.pt \
                 --include onnx \
                 --imgsz 640 \
                 --simplify

# Export sang TensorRT (cho thiết bị NVIDIA)
python export.py --weights best.pt --include engine --device 0
```

### 8.2. Lưu ý khi dùng DBB (Reparameterization)
**Quan trọng:** Trước khi export, phải gọi `reparameterize()` trên tất cả module DBB để hợp nhất các nhánh thành 1 conv duy nhất:

```python
model = torch.load('best.pt')
for module in model.modules():
    if hasattr(module, 'reparameterize'):
        module.reparameterize()
torch.save(model, 'best_reparam.pt')
```

---

## 9. CHECKLIST TRIỂN KHAI

- [ ] Thu thập ≥2000 ảnh đa điều kiện
- [ ] Cân bằng dữ liệu giữa các lớp (target ratio ≤ 10:1)
- [ ] Thực hiện tăng cường dữ liệu, đạt ≥6000 ảnh sau augmentation
- [ ] Thiết lập môi trường (Python 3.9, PyTorch 1.12.1, CUDA 11.2)
- [ ] Cài đặt baseline YOLOv5s, huấn luyện và ghi kết quả baseline
- [ ] Tích hợp từng cải tiến, chạy ablation study
- [ ] Đánh giá đầy đủ: mAP@0.5, mAP@0.5:0.95, $AP_s$, FPS
- [ ] Reparameterize DBB trước khi export
- [ ] Export ONNX/TensorRT cho deployment

---

## 10. TÀI LIỆU THAM KHẢO CHÍNH

1. Wang C. et al., *DP-YOLO: Effective Improvement Based on YOLO Detector*, Applied Sciences (MDPI), 2023.
2. DP-YOLO for Pest Detection (kích thước nhỏ, thiết bị nhúng) – bài báo nguồn.
3. Chen L. et al., *DP-YOLO: A Lightweight Real-Time Detection Algorithm for Rail Fastener Defects*, Sensors (MDPI), 2025.
4. Dai J. et al., *Deformable Convolutional Networks (DCNv2)*, CVPR 2019.
5. Wang X. et al., *InternImage: DCNv3*, CVPR 2023.
6. Ding X. et al., *Diverse Branch Block (DBB)*, CVPR 2021.
7. Zhu X. et al., *MPDIoU*, arXiv 2023.
8. Focaler-IoU, WIoU v3 – các bài báo mở rộng IoU loss.
