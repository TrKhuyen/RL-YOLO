> V0 giữ/bỏ đã triển khai và kiểm kỹ thuật31/10: [báo cáo/code/lệnh pilot](report_final/KB2_CONDITIONAL_V0_IMPLEMENTATION_20261031.md). Chưa chạy pilot2.000 update; phần dưới là lịch sử brainstorm.

> Bản tổng hợp và review mới ngày31/10/2026: [KB2_DPO_DESIGN_BRAINSTORM_20261031.md](KB2_DPO_DESIGN_BRAINSTORM_20261031.md). Phân biệt đề xuất V0 giữ/bỏ, các mở rộng chưa kiểm chứng và thiếu sót cần giải trước code; phần dưới giữ làm ghi chép brainstorm trước đó.

Mình đề xuất xây lại KB2 theo mục tiêu:
Dùng DPO để YOLO ưu tiên một bộ detection hoàn chỉnh tốt hơn trên cùng ảnh, với preference được xác định bằng cách so với GT.

Điểm khó cần giải quyết là biến YOLO thành model có thể sinh nhiều kết quả và tính xác suất của từng kết quả đó. Nếu chưa làm rõ phần này, rất dễ quay lại candidate ranking như hiện tại.
1. Chốt đơn vị đầu vào và đầu ra
Một mẫu huấn luyện sẽ có:
Input: một ảnh x

Output A:
  - Object 1: class + bounding box
  - Object 2: class + bounding box
  - ...

Output B:
  - Object 1: class + bounding box
  - ...

GT: các object và bounding box chuẩn của ảnh x
A và B có thể khác nhau về:
- Số object được phát hiện.
- Lớp của object.
- Vị trí và kích thước bounding box.
- Object bị bỏ sót hoặc dự đoán thừa.
A/B là dữ liệu dự đoán; hai ảnh vẽ nhãn chỉ phục vụ quan sát. Model học trên cấu trúc class/box, không học từ các pixel của đường bounding box đã vẽ.
2. Kiến trúc tổng thể nên như thế nào?
Mình đề xuất giữ YOLO làm mạng nền, thêm một cơ chế diễn giải output thành phân phối xác suất:
                       Ảnh train
                           │
                  YOLO đang học — policy
                           │
                Bộ giải mã có khả năng lấy mẫu
                      ┌────┴────┐
                      A         B
                      │         │
                      └────┬────┘
                           │
                    So với GT của ảnh
                           │
                  Chosen / Rejected
                           │
              Policy và reference chấm xác suất
                 của cùng Chosen / Rejected
                           │
                         DPO loss
                           │
                   Cập nhật policy
Policy và reference đều khởi tạo từ best checkpoint supervised tương ứng. Reference đóng băng.
Reference không phải “người tạo đáp án sai B”. A/B có thể đều được lấy mẫu từ cùng một model sinh; reference có vai trò chấm xác suất của cả hai để làm mốc trong DPO.
Để bắt đầu, mình nghiêng về sinh preference dataset cố định từ checkpoint ban đầu, rồi huấn luyện DPO trên dataset đó. Khi đã kiểm chứng cơ chế, mới xem xét làm mới cặp theo từng vòng.
3. Sinh hai bộ detection khác nhau bằng cách nào?
YOLO thông thường đưa ra tọa độ và score khá xác định. Chạy hai lần cùng ảnh chưa tạo ra hai đáp án có ý nghĩa.
Do đó, cần thiết kế một stochastic decoder — bộ giải mã có lấy mẫu.
Một phương án đáng nghiên cứu là diễn giải dense output theo các quyết định:
Quyết định	Ý nghĩa
Giữ hoặc không giữ candidate	Có xuất một detection tại vị trí này không?
Chọn lớp	Nếu xuất detection, object thuộc lớp nào?
Chọn tọa độ	Bounding box cụ thể là gì?


Từ cùng ảnh và cùng mạng, lấy mẫu hai lần các quyết định đó để có A và B.
Với tọa độ, có thể nghiên cứu một phân phối quanh tọa độ YOLO dự đoán, dùng độ phân tán cố định ở bản thử nghiệm đầu. Phân phối cụ thể và cách bảo đảm hộp hợp lệ vẫn cần thiết kế, chưa nên mặc định thêm Gaussian noise là xong.
Điều quan trọng là:
Cách lấy mẫu phải đi kèm công thức tính xác suất. Nhiễu chỉ để tạo hai hình khác nhau chưa đủ cho DPO.

Mình không đề xuất dùng hai ảnh augmentation khác nhau làm hai “đáp án”. Nên giữ cùng input để chất lượng A/B phản ánh khác biệt output, tránh trộn thêm độ khó của ảnh.
4. Dùng GT chọn A hay B như thế nào?
Nên có một bộ chấm toàn bộ detection set, trả lời ba câu:
1. Tìm đúng được bao nhiêu object?
2. Có bỏ sót hoặc dự đoán thừa không?
3. Các hộp đúng khít GT đến mức nào?
Trước tiên matching prediction với GT theo đúng lớp và một-một. Một GT không được tính đúng nhiều lần vì có nhiều hộp trùng.
Một công thức khởi đầu để thảo luận:
\[
Q(S,GT)=w_d\,D(S,GT)+w_l\,L(S,GT)
\]
Trong đó:
- \(D\): chất lượng phát hiện, có phạt FP và FN, chẳng hạn F1.
- \(L\): chất lượng định vị, chẳng hạn tổng IoU của các cặp matched chia cho số GT.
- \(w_d,w_l\): trọng số thiết kế cần khóa và kiểm tra.
Chuẩn hóa localization theo số GT giúp bộ chỉ phát hiện một object dễ không được đánh giá quá cao vì hộp đó rất khít.
Ví dụ:
	A	B
GT trong ảnh	3	3
Phát hiện đúng	3	2
Bỏ sót	0	1
Dự đoán thừa	0	1
Hộp matched	Khá khít	Khá khít
Preference	Chosen	Rejected


Những trường hợp A nhiều TP hơn nhưng box kém khít hơn B cần công thức xử lý minh bạch. “Tốt hơn” phải được định nghĩa trước khi train, không đổi cách chấm sau khi thấy kết quả.
Cũng nên có ngưỡng gần hòa: nếu hai bộ gần tương đương thì bỏ cặp đó, không ép tạo nhãn ưu tiên nhiễu.
5. Tính xác suất bộ detection: phần quyết định thiết kế có đúng DPO không
Với output đã chọn, cần tính:
\[
\log \pi_\theta(S\mid x)
\]
Nếu thiết kế output bằng các quyết định tại những vị trí cố định, có thể xây joint log-probability từ:
Log-probability của các quyết định giữ/bỏ
+
Log-probability lớp của các detection được giữ
+
Log-density tọa độ của các detection được giữ
Khi đó cả policy và reference đều chấm cùng quyết định, cùng lớp, cùng tọa độ đã lưu. Policy không sinh lại một bộ hộp mới thay cho A khi đang tính xác suất A.
Có một giới hạn phải ghi rõ: likelihood của một cấu trúc gắn với candidate indices chưa tự động là likelihood của tập hộp không có thứ tự. Nhiều cấu trúc candidate có thể tạo cùng một tập hộp. Bản đầu có thể định nghĩa action là cấu trúc có chỉ số và chấm chất lượng toàn bộ detection mà nó tạo ra, nhưng phải gọi tên chính xác.
Tương tự, NMS có thể biến nhiều output thành cùng kết quả cuối. Không được lấy log-probability trước NMS rồi mặc định đó là xác suất chính xác của tập hộp sau NMS.
Mình ưu tiên một policy có xác suất rõ ràng trước, sau đó mới mở rộng độ phức tạp.
6. DPO loss và native detection loss nên dùng thế nào?
Khi đã có chosen \(S^+\), rejected \(S^-\), DPO dùng:
\[
L_{\text{DPO}}
=
-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(S^+\mid x)}{\pi_{\mathrm{ref}}(S^+\mid x)}
-
\log\frac{\pi_\theta(S^-\mid x)}{\pi_{\mathrm{ref}}(S^-\mid x)}
\right]\right)
\]
Đây là phần kế thừa từ DPO gốc; định nghĩa policy detection và bộ chấm GT là phần nghiên cứu của KB2.
Mình đề xuất:
- Ở bước kiểm chứng cơ chế: dùng DPO-only để kiểm tra gradient có tăng ưu tiên chosen và đi được qua classification/regression như dự định hay không.
- Ở bước huấn luyện detector: đánh giá có cần thêm native loss để giữ năng lực nhận diện hay không.
Nếu cần:
\[
L_{\text{total}}=L_{\text{DPO}}+\lambda_{\text{det}}L_{\text{native}}
\]
Đó vẫn là một phương pháp DPO có thành phần giữ năng lực tác vụ. Tuy nhiên, chưa nên chốt \(\lambda_{\text{det}}\) trước khi biết DPO-only hoạt động ra sao.
7. Phiên bản đầu nên thu gọn đến đâu?
Mình đề xuất bắt đầu bằng một model YOLOv8n, với:
Thành phần	Thiết kế ban đầu
Khởi tạo	Best supervised checkpoint chung
Reference	Bản sao đóng băng của checkpoint đó
Input A/B	Cùng ảnh, cùng transform
Output	Hai bộ detection được lấy mẫu
Preference	Chấm toàn bộ output với GT train
Dataset preference	Cố định trong vòng thử nghiệm đầu
Thành phần được học	Bắt đầu từ detection head để kiểm soát thay đổi
Đối chiếu chính	Checkpoint sau DPO với supervised checkpoint
Selection	Validation, cùng evaluator cho hai checkpoint


Không cần mở ngay sang năm model, online refresh, hay nhiều công thức reward. Trước tiên phải chứng minh được A/B khác nhau có ý nghĩa, likelihood đúng và DPO thực sự tác động lên quyết định detection.
Thiết kế này bám ý tưởng của bạn hơn bản hiện tại. Phần cần đào sâu tiếp theo là stochastic decoder và biểu diễn action: đây là nơi quyết định KB2 có trở thành DPO cho bộ detection một cách nhất quán hay chỉ là một loss ranking khác.

Mình đề xuất một thiết kế chính cho KB2, giải quyết cả ba câu hỏi và bộ metric đánh giá. Đây vẫn là phương án nghiên cứu, chưa phải phương pháp đã được kiểm chứng.
Điểm xuất phát giữ đúng điều đã chốt: policy và reference đều lấy từ best checkpoint supervised KB1; chỉ policy được cập nhật.
1. Sinh hai bộ detection: lấy mẫu từ một policy được định nghĩa trên output YOLO
Không gọi predict() hai lần theo cách thông thường. Thay vào đó:
Cùng một ảnh
      ↓
YOLO tính đặc trưng và output head
      ↓
Bộ giải mã xác suất
      ├── Lấy mẫu lần 1 → bộ detection A
      └── Lấy mẫu lần 2 → bộ detection B
Mình nghiêng về tận dụng phân phối định vị sẵn có của YOLOv8/YOLO11 cho bản đầu.
Ở các head dùng DFL, mỗi phía của bounding box được biểu diễn bằng phân phối trên các bin khoảng cách. Decoder thông thường tính giá trị kỳ vọng của các phân phối này để tạo tọa độ. Đây là hành vi thể hiện trong module DFL chính thức của Ultralytics.
Đề xuất của KB2 là nghiên cứu lấy mẫu các khoảng cách từ những phân phối đó, thay vì luôn lấy kỳ vọng.
Ví dụ minh họa cho một phía hộp:
Khoảng cách	Xác suất
4	0,15
5	0,60
6	0,25


Hai lần lấy mẫu có thể chọn 5 và 6. Khi lấy mẫu cả bốn phía, ta nhận được hai hộp khác nhau.
Tuy nhiên, chỉ lấy mẫu tọa độ chưa đủ. A/B cần có khả năng khác nhau về lớp và số lượng detection. Vì vậy, policy mới cần ba loại quyết định:
Quyết định	Cách biểu diễn đề xuất
Có xuất detection tại candidate này không?	Có lựa chọn background/không xuất
Nếu xuất, thuộc lớp nào?	Phân phối lớp
Bounding box là gì?	Phân phối khoảng cách bốn phía


Với classification, có thể xây một phân phối chuẩn hóa gồm 28 lớp + background, từ class logits và một quy tắc background được định nghĩa rõ. Đây là cách mô hình hóa mới của KB2, vì sigmoid scores hiện tại của YOLO chưa tự là một categorical distribution có background.
Lợi ích của cách này: A/B có thể khác nhau về số object, lớp và tọa độ; đồng thời mỗi quyết định lấy mẫu có xác suất để tính DPO loss.
Cần giữ hai quy tắc:
- GT không tham gia sinh A/B. GT chỉ chấm sau khi hai output được tạo.
- Policy và reference phải tính xác suất của đúng các quyết định A/B đã lưu, không tự sinh lại hộp thay thế.
Ở bản đầu, mình đề xuất tạo một preference dataset cố định từ model supervised. Có thể sinh vài output cho mỗi ảnh, sau đó giữ các cặp khác nhau và có chênh lệch chất lượng rõ. Chưa cần refresh online ngay.
Có một giới hạn phải gọi tên chính xác: action ban đầu sẽ là toàn bộ cấu trúc quyết định theo candidate indices, tạo ra một bộ detection. Xác suất của cấu trúc đó chưa đồng nhất với xác suất của một tập hộp không thứ tự sau NMS. Phần decode/NMS phải được khóa, không thay đổi ý nghĩa của A/B khi policy cập nhật.
Vì vậy, nên bắt đầu với YOLOv8n. YOLO26 không có cùng cơ chế DFL nên cần thiết kế phân phối tọa độ riêng; chưa nên ép tất cả model vào một công thức.
2. Dùng GT chọn A/B: chấm cả bộ detection, có phạt bỏ sót và dự đoán thừa
Mình đề xuất bộ chấm gồm hai phần:
1. Phát hiện đúng object: đúng lớp, không bỏ sót nhiều, không sinh nhiều hộp thừa.
2. Định vị tốt: các hộp đúng cần khít GT.
Trước tiên, matching prediction với GT phải:
- Đúng lớp.
- Một-một.
- Có quy tắc xử lý xung đột xác định.
- Không cho nhiều prediction cùng nhận công một GT.
Một công thức khởi đầu hợp lý là:
\[
Q(S,GT)=0.8\,\overline{F_1}(S,GT)+0.2\,I_{\text{GT}}(S,GT)
\]
Trong đó:
\[
\overline{F_1}
=
\operatorname{mean}_{t\in\{0.50,0.55,\ldots,0.95\}}F_{1,t}
\]
\[
F_{1,t}=\frac{2TP_t}{2TP_t+FP_t+FN_t}
\]
Còn:
\[
I_{\text{GT}}
=
\frac{\sum_{\text{cặp matched tại IoU}\ge0.5}IoU}{N_{\text{GT}}}
\]
Các trọng số 0,8/0,2 là đề xuất khởi đầu để kiểm tra, chưa phải giá trị tối ưu.
Tại sao dùng cách này?
- Đúng lớp nhưng hộp lệch sẽ mất điểm ở các ngưỡng IoU cao.
- Bỏ sót làm tăng FN.
- Dự đoán thừa, trùng hoặc sai lớp làm tăng FP; sai lớp còn có thể để lại FN.
- Chia tổng IoU cho số GT giúp bộ chỉ tìm được một object dễ không được đánh giá quá cao.
Ví dụ:
Kết quả	A	B
GT trong ảnh	3	3
Phát hiện đúng	3	2
Bỏ sót	0	1
Hộp thừa	0	1
Độ khít hộp đúng	Khá tốt	Khá tốt
Preference	Chosen	Rejected


Nếu A tìm đủ object nhưng hộp kém khít hơn B, công thức sẽ thể hiện đánh đổi đó một cách nhất quán.
Chọn preference bằng:
\[
Q(A)-Q(B)>\delta
\Rightarrow A\text{ là chosen}
\]
Nếu chênh lệch nhỏ hơn ngưỡng \(\delta\), bỏ cặp gần hòa. Không ép mọi ảnh phải tạo được preference.
Ảnh không có GT cần quy tắc riêng: ưu tiên output rỗng và giảm điểm theo số FP. Không đưa trường hợp này vào công thức có mẫu số \(N_{\text{GT}}\) mà chưa xử lý.
Mình không đề xuất lấy confidence làm tiêu chí quyết định bộ nào đúng hơn. Confidence là thứ model cần học cho phù hợp; GT mới là căn cứ chấm chất lượng.
3. DPO và native detection loss: dùng chung trong phương pháp chính
Sau khi có chosen \(a^+\) và rejected \(a^-\), tính:
\[
L_{\text{DPO}}
=
-\log\sigma\left(
\beta\left[
\log\frac{\pi_\theta(a^+\mid x)}{\pi_{\text{ref}}(a^+\mid x)}
-
\log\frac{\pi_\theta(a^-\mid x)}{\pi_{\text{ref}}(a^-\mid x)}
\right]\right)
\]
Đây là cấu trúc objective của DPO gốc. Policy detection và cách tạo preference là phần thiết kế của KB2.
Log-probability phải tính từ các quyết định đã lấy mẫu, gồm cả:
- Class/background ở các candidate.
- Các bin tọa độ của detection được xuất.
Không chỉ cộng confidence của các hộp được giữ rồi bỏ qua quyết định không xuất hộp.
Với bài toán hiện tại, mình khuyên phương pháp huấn luyện chính dùng:
\[
\boxed{L_{\text{total}}=L_{\text{native}}+\lambda L_{\text{DPO}}}
\]
Vai trò của hai thành phần:
Thành phần	Nhiệm vụ
Native detection loss	Giữ khả năng học lớp và định vị trực tiếp từ GT
DPO loss	Điều chỉnh phân phối để ưu tiên cấu trúc dự đoán tạo ra bộ detection tốt hơn
Reference	Làm mốc trong objective DPO


Lý do mình chọn kết hợp: DPO chỉ được biết “A tốt hơn B”. A vẫn có thể còn sai nhiều. Native loss cung cấp thêm đích sửa cụ thể từ GT.
DPO không cần đạo hàm xuyên bước lấy mẫu: sau khi lưu action, gradient đi qua log-probability của action đó. Với thiết kế trên, gradient DPO có thể tác động cả classification và phân phối regression.
Đây vẫn là một phương pháp DPO của KB2, không cần quay lại năm nhánh cũ.
Về tham số:
- \(\lambda\) điều chỉnh sức mạnh DPO so với native.
- \(\beta\) điều chỉnh thang relative log-ratio.
- Temperature lấy mẫu điều chỉnh độ đa dạng A/B; nó khác \(\beta\).
Chưa nên kế thừa máy móc alpha=beta=0.1 của code cũ vì action mới lớn hơn nhiều. Cần đo phân bố log-ratio và gradient hai loss để chọn một nhóm cấu hình pilot nhỏ.
Có thể dùng DPO-only làm kiểm tra cơ chế hoặc ablation. Nhưng mình chưa chọn nó làm cấu hình huấn luyện chính trước khi có bằng chứng giữ được chất lượng detector.
4. Metric đánh giá: cần đo detector và cơ chế DPO riêng
Chất lượng detector — đây là kết quả chính của đề tài
Metric	Mục đích
mAP50–95	Metric chính, chọn best checkpoint
mAP50	Khả năng phát hiện ở ngưỡng định vị tương đối rộng
mAP75	Theo dõi chất lượng định vị khắt khe hơn
AP từng lớp	Phát hiện lớp bị cải thiện hoặc suy giảm
Precision, Recall, F1	Đo hành vi tại một operating threshold đã khóa
FP/ảnh và FN rate	Kiểm tra dự đoán thừa và bỏ sót
Mean matched IoU + tỷ lệ GT matched	Giải thích độ khít, tránh chỉ nhìn các object dễ
AP theo kích thước	Kiểm tra object nhỏ/vừa/lớn khi có đủ support


mAP50–95 dùng các ngưỡng IoU từ 0,50 đến 0,95; AP/AR và giới hạn detections cần theo một contract evaluator đã khóa. COCOeval chính thức.
Precision/Recall/F1 phải ghi rõ confidence và IoU matching. Không dùng COCO average recall thay cho recall tại một operating point.
Matched IoU phải đi kèm coverage: model bỏ hết object khó có thể tăng mean IoU trên vài object còn lại nhưng detector thực tế kém hơn.
Cơ chế preference — để biết DPO có học đúng điều mong muốn không
Diagnostic	Câu hỏi cần trả lời
Tỷ lệ A/B khác nhau sau decode/NMS	Sampler có tạo đa dạng hữu ích không?
Tỷ lệ cặp được giữ, cặp gần hòa	Có đủ preference đáng tin cậy không?
Phân bố \(Q_{\text{chosen}}-Q_{\text{rejected}}\)	Cặp quá dễ, quá khó hay gần hòa?
Policy/reference log-probability gaps	Model có dịch ưu tiên đúng hướng không?
DPO loss trên bộ preference audit cố định	Objective có được tối ưu qua thời gian không?
Gradient norm/cosine native–DPO	Hai thành phần có quá yếu hoặc xung đột không?
Chất lượng trung bình của các output lấy mẫu mới	Phân phối sinh kết quả có tốt lên không?
Số detection và tỷ lệ output rỗng	Có suy biến thành bỏ hết hoặc sinh quá nhiều hộp không?


Không dùng “chosen có Q cao hơn rejected” làm bằng chứng model đã học: điều đó đúng ngay từ lúc gán nhãn cặp.
Hiệu quả thực nghiệm
Báo thêm:
- Chênh lệch AP so với supervised checkpoint.
- Mean và SD qua các continuation seed.
- Thời gian train, VRAM và chi phí sinh preference.
- Latency inference đo riêng nếu cần tuyên bố tốc độ.
Đặc biệt, đánh giá chính phải dùng cách inference triển khai đã chọn và áp dụng giống nhau cho supervised lẫn model sau DPO. Nếu sampler tốt lên nhưng inference YOLO thông thường không tốt lên, chưa thể kết luận detector triển khai được cải thiện.
Mình đề xuất bắt đầu bằng YOLOv8n, preference dataset cố định, DPO kết hợp native loss, với mAP50–95 là metric chính và các diagnostic trên để kiểm chứng cơ chế. Trước khi train dài, cần xác nhận ba điều: A/B khác nhau có ý nghĩa, log-probability được tính đúng, và gradient DPO đi được vào cả quyết định lớp lẫn định vị.
