# KB2 — Tổng hợp brainstorm và review thiết kế, 31/10/2026

> Cập nhật sau brainstorm: [Conditional V0 đã chạy dài15k với đối chứng](report_final/KB2_CONDITIONAL_LONG_REVIEW_20261031.md). Recipe không giảm FN ở confidence0,25; matched-FP có gain nhỏ ở cả native/DPO. Nội dung bên dưới giữ phạm vi thiết kế và lịch sử quyết định.

**Cập nhật31/10: V0 giữ/bỏ đã triển khai và kiểm kỹ thuật trên GPU; chưa chạy pilot2.000 update hoặc train dài.** Xem [báo cáo triển khai](report_final/KB2_CONDITIONAL_V0_IMPLEMENTATION_20261031.md). Các mở rộng class/NMS/Gaussian trong tài liệu vẫn là đề xuất. Bản brainstorm gốc: Tài liệu tổng hợp trao đổi sau audit ngày30/10; không thay thế kết quả thực nghiệm cũ và không tuyên bố phương pháp đã cải thiện. Các giá trị pilot bên dưới là đề xuất để khóa trước khi chạy.

Hướng nên thử trước: **DPO có điều kiện cho quyết định giữ/bỏ candidate, có cả giữ detection đúng và loại prediction sai**. Chấm tác động trên toàn ảnh, dùng S0 làm khởi tạo/reference/mốc so sánh. Sửa class, cụm candidate tương tác và policy tọa độ là các bước mở rộng được kiểm tra riêng.

## 1. Những gì đã có bằng chứng

- [Thử F1→F2](report_final/KB2_F2_EXPERIMENT_20261028.md): ba seed chỉ giảm trung bình0,67FN/run so recipe F1; chưa có bằng chứng đổi scorer giải quyết bỏ sót.
- [Recovery long](report_final/KB2_RECOVERY_LONG_REVIEW_20261030.md): tại confidence0,25, S0 FN867; F2 best seed42 FN920; recovery best FN929. Recovery có47,78% lượt cặp khôi phục, nhưng chưa cải thiện mục tiêu FN.
- [Audit tại GT](report_final/KB2_DPO_GT_ALIGNMENT_AUDIT_20261030.md): trong328GT được chosen cứu so rejected,170GT vốn được ordinary S0 phát hiện. Trên241 cặp khôi phục thuộc DPO train, best margin trung bình+3,3608; phần trực tiếp tại slot cứu GT đóng góp âm, các slot còn lại đóng góp dương.
- Trên143GT mục tiêu thuộc DPO train thực sự bị S0 bỏ sót, score tại slot nguồn cố định giảm trung bình0,149083→0,125160; IoU box ordinary tăng nhẹ0,793352→0,798702. Best cứu10GT trong nhóm143 này, nhưng mất38/152GT mục tiêu S0 từng phát hiện trong cùng tập audit295GT.
- Native loss vẫn được dùng trong các run. So lịch sử F2/recovery chưa tách mọi yếu tố; chưa được gán toàn bộ thay đổi cho riêng DPO. Audit gradient và vài bước AdamW chưa thay thế thí nghiệm huấn luyện có đối chứng.

Các kết quả này chỉ ra những giả thuyết cần sửa; chưa chứng minh thiết kế mới chắc chắn tốt hơn.

## 2. Mục tiêu và phạm vi cần gọi đúng

Giữ mục tiêu nghiên cứu: dùng preference từ GT train để YOLO ưu tiên detection hữu ích hơn, giảm FN có kiểm soát FP và giữ chất lượng định vị. Policy/reference khởi tạo từ best supervised KB1 tương ứng, gọi là S0; reference đóng băng.

**Đề xuất mới thu hẹp action được học trong mỗi cặp.** A/B được render thành hai bộ detection toàn ảnh, nhưng likelihood là xác suất của quyết định sửa có điều kiện trên phần còn lại. Đây không phải likelihood biên của tập box không thứ tự sau NMS, cũng không đồng nhất với policy whole-action hiện tại.

Nếu đề tài yêu cầu tối ưu likelihood của cả bộ detection vô điều kiện theo ảnh, đề xuất này chỉ là một bước thử nghiệm về quyết định cục bộ, chưa đáp ứng trọn phạm vi đó. Không dùng cách gọi “DPO toàn detection set” để che khác biệt.

Không thêm mạng reward hoặc head mới trong pilot giữ/bỏ. Không sửa ảnh/annotation. Ordinary inference trên ảnh mới không nhận GT, danh sách FN, reference hay phần dùng chung.

## 3. Các khái niệm

| Khái niệm | Định nghĩa |
|---|---|
| GT mục tiêu | Một object có ID ổn định từ ảnh nguồn và dòng annotation |
| Candidate/slot | Một vị trí đầu ra YOLO có class logits và box; nhiều slot có thể dự đoán cùng một lá |
| Quyết định | Trường đầu ra đem ra so sánh: giữ/bỏ, class hoặc tọa độ |
| Phần chung C | Mọi quyết định được giữ giống nhau giữa A/B trước render/NMS; có thể vẫn có TP, FP và FN |
| Phần riêng | Một quyết định hoặc một cụm nhỏ các quyết định khác nhau giữa A/B |
| Chosen/rejected | Kết quả sau GT judge, không mặc định A tốt hơn B |

“Chung/cố định” chỉ áp dụng trong cặp dữ liệu. Nó không có nghĩa đã đúng, không có nghĩa model không được thay đổi score ở đó, và không có nghĩa sau NMS hai output vẫn giống nhau ngoài vùng sửa.

YOLO hiện tại nhận ảnh, không nhận C như một prefix của LLM. C xác định các biến được giữ cố định và cách render; scorer truy vấn logits tại candidate liên quan. Không tuyên bố model đã học suy luận phụ thuộc đầy đủ vào danh sách detection C.

## 4. Pipeline đề xuất

```text
S0 + ảnh train → ordinary prediction + raw candidates
             → đối chiếu GT và phân loại lỗi
             → chọn candidate/quyết định có thể tạo cặp
             → lưu C và hai lựa chọn khác nhau
             → render toàn ảnh, NMS cố định, GT judge
             → giữ cặp rõ ràng + thông tin độ bao phủ

Replay cùng ảnh/transform/action/C
             → policy và reference chấm cùng lựa chọn
             → native loss + lambda DPO loss
             → kiểm tra ordinary detection và GT cứu/mất

Triển khai: ảnh mới → policy YOLO → ordinary inference
```

Bản đầu dùng offline pairs cố định từ S0 và transform xác định. Không chạy hai lần predict thông thường rồi giả định tự có hai action khác nhau. Bộ sinh dùng các đề xuất/lựa chọn có điều kiện; không tuyên bố chúng là hai mẫu độc lập vô điều kiện của model.

## 5. Chọn lỗi và tạo cặp

Chạy ordinary S0 trên train với protocol canonical: input640, inference confidence0,001, NMS0,60, max_det300; operating confidence0,25/IoU0,5. Ghi ID GT matched/unmatched và prediction TP/FP theo matcher đã khóa. Sau đó xem raw candidates để tìm ứng viên sửa.

| Loại | Chọn ứng viên | Chosen/rejected | Điểm cần tránh |
|---|---|---|---|
| Giữ detection đúng | FN thật của ordinary S0 có candidate đúng class/box tốt nhưng score thấp; bổ sung GT đã đúng để giữ năng lực | Giữ / bỏ cùng candidate | Chosen không được ghép box GT hoặc nâng score bằng tay |
| Loại prediction nền sai | Candidate không liên quan hợp lý tới GT nào và có cặp drop giảm FP | Bỏ / giữ | Prediction unmatched chưa đủ để kết luận là nền |
| Loại duplicate | Candidate dư cho object đã có detection khác | Bỏ / giữ | Sau drop vẫn phải giữ GT đó qua matching một-một |
| Sửa class | Box phù hợp một GT nhưng class sai | Class đúng / class sai, cùng box | Giảm class sai có thể chưa nâng class đúng qua ngưỡng |
| Sửa box | Candidate liên quan GT nhưng định vị kém | Tọa độ tốt / kém, cùng class | Bin/sample tốt hơn không bảo đảm ordinary box tốt hơn |

Box IoU0,49 với lá thật có thể là lỗi định vị; prediction đúng vùng lá nhưng sai class là lỗi class. Không đưa máy móc chúng vào negative-background. Những trường hợp nhập nhằng được bỏ hoặc đưa sang nhóm sửa tương ứng.

V0 chỉ dùng giữ/bỏ với box cố định. Hai hướng preference cùng tồn tại; không suy rằng cân bằng số lượng sẽ cân bằng gradient. Ngưỡng chất lượng phụ, cách lấy candidate, tỷ lệ FN/GT đã đúng và cách phân loại negative phải được ghi trước audit và công bố độ bao phủ, không chọn theo test.

### Ví dụ ảnh có sáu lá

Giả sử S0 đúng lá1/2, lá3 có candidate tốt nhưng score thấp, lá4 nhầm class, lá5/6 bị bỏ sót, và có prediction nền P. Cặp nhắm lá3:

| Thành phần | A | B |
|---|---|---|
| Lá1/2 | Detection chung | Giống A |
| Candidate lá3 | Giữ | Bỏ |
| Lá4 | Cùng class đang sai | Giống A |
| Lá5/6 | Cùng chưa được phát hiện | Giống A |
| P | Cùng prediction sai | Giống A |

GT có thể chọn A nếu nó cứu đúng lá3, không tăng lỗi khác. Cặp khác nhắm P sẽ chọn drop P. Native loss học toàn bộ GT trên ảnh. Không cần crop riêng từng lá; một ảnh có thể cung cấp nhiều cặp nhưng bị giới hạn lặp.

### Gate trên toàn bộ output

1. GT judge đánh giá hai output sau render/NMS và matching một-một.
2. Cặp cứu GT phải cứu đúng ID mục tiêu; không mất GT rejected đã tìm được và không tăng FP.
3. Cặp drop phải giảm lỗi thực tế; không làm mất GT đang được phát hiện. Nếu candidate khác thay thế khiến FP không giảm, không tự nhận đó là cặp loại FP hữu ích.
4. Ngoài trường được phép thay, các action fields phải giống nhau chính xác trước NMS. Sau NMS có thể phát sinh khác biệt rộng hơn; phải ghi lại và chấm toàn ảnh.
5. Cặp class/box có điều kiện chất lượng riêng, gồm khoảng cách chất lượng tối thiểu và bảo vệ GT khác. Chưa triển khai trong V0.

Không bắt mọi cặp cứu một GT ở ảnh nhiều lá phải vượt cùng một minimum-gap của điểm F2 toàn ảnh. F2 có thể dùng làm diagnostic; gate nhắm đúng lỗi là chính. Strict gate có thể thiên về ảnh dễ: báo tỷ lệ cặp hợp lệ theo class, kích thước object và mật độ/chồng lấn lá.

**Lỗ hổng cần tránh:** confidence của candidate cứu FN có thể vẫn dưới0,25 lúc sinh cặp. Action keep cho phép nó xuất hiện để so sánh cấu trúc, không tự gán score cao. Judge preference và ordinary evaluation sau update có phạm vi confidence khác nhau và phải ghi rõ. Không ép A đạt0,25 ngay lúc tạo cặp, vì sẽ loại chính FN confidence đang nghiên cứu.

## 6. Policy và loss V0

Với candidate j, class c và box b_C cố định trong C, định nghĩa policy nhị phân chuẩn hóa:

\[
\pi_\theta(a=1\mid x,C)=\sigma(z_{\theta,j,c}),\qquad
\pi_\theta(a=0\mid x,C)=1-\sigma(z_{\theta,j,c}).
\]

Sigmoid ở đây là tham số Bernoulli được chọn cho action giữ/bỏ; không mặc định là xác suất đã hiệu chỉnh của một detection đúng sau NMS. Giữ temperature1 ở V0 để log-odds bằng raw logit.

Cho y=+1 nếu GT thích keep, y=-1 nếu GT thích drop. DPO với reference S0 trở thành:

\[
m=y(z_{\theta,j,c}-z_{0,j,c}),\qquad
L_{DPO}=\operatorname{softplus}(-\beta m).
\]

Đây là DPO trên policy nhị phân nêu trên; về đại số là logistic classification với offset reference. Cấu trúc loss tham chiếu [DPO gốc](https://arxiv.org/html/2305.18290v3), còn policy detection này là đề xuất mới. Nó không tự chứng minh lợi ích riêng so supervised hard-example learning. Không chỉ bỏ term DFL khỏi log-prob joint cũ rồi gọi đó là cùng policy.

\[
L_{total}=L_{native}+\lambda L_{DPO}.
\]

Policy/reference phải chấm cùng ảnh, transform, slot/class, box C và lựa chọn được lưu, cùng batch shape/precision. Khi score action, không sinh lại A/B hoặc thay box cũ bằng box policy hiện tại. Initial margin phải bằng0 trong sai số số học.

Nếu sau này dùng nhóm candidate, các action và xác suất joint/conditional phải được định nghĩa có chuẩn hóa. Không nhân thêm “GT score” vào log-prob và coi đó là likelihood. Reference đóng băng không phải là model chuyên sinh rejected.

## 7. Bổ sung sau review: box lưu cố định nhưng regression có thể trôi

Nếu C lưu box tốt b_C từ S0 trong khi native tiếp tục train regression, raw score tại slot j có thể tăng nhưng box ordinary hiện tại b_theta,j đã khác b_C. Tăng xác suất keep một action có box cũ không chứng minh detection hiện tại tốt hơn. Đây là một mismatch mới cần đo.

**Đề xuất kiểm chứng V0 sạch hơn:** chỉ train nhánh classification của Detect head (cv3 trong adapter/model đang dùng); freeze regression cv2, integral DFL, backbone/neck và BN running statistics. Với cùng input/batch/precision, ordinary raw boxes phải không đổi. Thay đổi box trong trường hợp này là lỗi triển khai/numerical-protocol cần điều tra.

Raw boxes không đổi không có nghĩa detection set sau NMS không đổi: score/class thay đổi vẫn có thể đổi argmax, thứ tự ưu tiên, suppression và GT matched. Vẫn phải đo FP/FN toàn ảnh.

Hạn chế được ghi rõ: V0 kiểm tra classification/keep-drop, chưa phải DPO tối ưu box hoặc khả năng biểu diễn của backbone. Mở regression/native hoặc mở thêm backbone là thí nghiệm sau, với đối chứng có cùng trainable mask. Không đổi mask theo từng ảnh một cách ngầm định.

## 8. Native stream và cấu hình pilot đề xuất

Một cấu hình đáng thử để bảo vệ các lá khác trong ảnh có cặp:

- Batch4 gồm2 ảnh lấy đều từ full train và2 ảnh có preference.
- Native loss tính trên cả4 ảnh và toàn bộ GT của chúng. DPO tính trên phần có preference.
- Positive/negative pair sampling cân bằng khi có đủ cặp hợp lệ; có giới hạn lặp theo ảnh/GT/class và báo phân phối thực tế.
- Cùng chuỗi ảnh, cặp, seed, transform và trainable mask giữa các đối chứng; điều kiện native-only vẫn nhận đúng những ảnh đó.

Lợi ích kỳ vọng: khi DPO nhắm một candidate, native cùng ảnh cung cấp tín hiệu cho các lá khác. Đây vẫn là giả thuyết, không bảo đảm mọi GT được giữ lại. Native stream trộn cũng đổi phân phối exposure so run lịch sử, nên kết quả cũ chỉ là tham khảo; không dùng chúng làm đối chứng nhân quả trực tiếp.

| Tham số | Đề xuất khởi đầu | Giới hạn |
|---|---:|---|
| Model/khởi tạo | YOLOv8n/S0 | Một model trước khi nhân rộng |
| Trainable mask V0 | Classification cv3 | Thay đổi so Detect-head-only cũ |
| Input/batch | 640/4 | Native4, DPO chỉ phần có cặp |
| AdamW LR/weight decay | 1e-6/5e-4 | Cùng optimizer trong đối chứng |
| Clip norm | 1 | Log norm và clipping |
| beta/lambda | 0,1/0,01 | Chỉ điểm xuất phát; không suy là DPO chiếm1% tác động |
| Pilot | 2.000 update cố định | Eval0/1k/2k; lưu cảbest/last, S0 là ứng viên |
| Pair bank | Offline từ S0 | Transform xác định, không refresh ở V0 |

Các tỷ lệ30/30/20/20 và ngân sách3slot/8box từng được nêu chỉ là gợi ý brainstorm, chưa phải cấu hình đã chọn. V0 chưa cần sinh biến thể tọa độ. Trước code cần khóa candidate thresholds, cap lặp, tỷ lệ FN/TP-preservation và cách xử lý khi negative quá ít; không ép ratio bằng cách lặp vài mẫu vô hạn.

## 9. Mở rộng NMS và class

Một candidate có thể ảnh hưởng lá bên cạnh qua NMS. Nếu single-candidate gate thất bại do tương tác, thử cụm nhỏ candidate có overlap/quan hệ suppression, giữ phần còn lại chung và kiểm tra toàn ảnh. Không coi một lá bằng một slot; không loại hết ảnh đông lá để làm kết quả đẹp.

Giới hạn kích thước nhóm và số đề xuất để kiểm soát chi phí. Chưa khóa cách xây graph/tối đa nhóm: đó là thiếu sót phải giải trước khi chạy biến thể này. NMS chỉ dùng render/judge; gradient DPO không cần xuyên qua NMS. Giữ thuật toán/threshold NMS như S0 để đo DPO; Soft-NMS là hướng riêng nếu cần, không đổi đồng thời.

Class repair dùng categorical chuẩn hóa trên các class ứng viên cố định. Pair gap có thể tăng do giảm class sai mà true-class score vẫn chưa vượt ngưỡng. Vì vậy cần đo cả true-class score, class gap, argmax và ordinary recovery, không chỉ loss/margin.

## 10. Policy tọa độ: hai lựa chọn cần so riêng

### 10.1 DFL sampling hiện tại

Tận dụng phân phối model có sẵn, nhưng tăng xác suất sampled bins không bảo đảm expectation/ordinary box tốt hơn. Gate sampled-IoU cần kèm đo hướng thay đổi ordinary-IoU, finite difference và bước AdamW. Không mặc định DFL probabilities là uncertainty đã hiệu chỉnh.

### 10.2 Gaussian quanh ordinary box — đề xuất thay thế

Định nghĩa action u ở tọa độ tâm chuẩn hóa và log kích thước:

\[
u=(c_x/W,c_y/H,\log(w/W),\log(h/H)),\qquad
q_\theta(u\mid x,j,C)=\mathcal N(u;\mu_\theta(x,j),\Sigma_C).
\]

mu_theta là biểu diễn của ordinary decoded box. Sigma_C cố định, dương xác định, dùng giống nhau ở policy/reference và được lưu cùng cặp; chưa học variance. Tọa độ log kích thước giữ width/height dương. Chấm log-density của action u trước clip/NMS, không nhận đó là density của box sau clipping hoặc tập sau NMS.

Đề xuất cặp có thể dùng mu_generator ± delta rồi GT chọn; hai lựa chọn có thể tương quan. Khi replay, u+,u- và covariance giữ nguyên; chỉ mu_policy thay đổi. Việc đề xuất/loại cặp không được âm thầm thay likelihood thành một phân phối truncated khác chưa chuẩn hóa.

**Hạn chế quan trọng:** Gaussian không tự ngăn overshoot. Ví dụ1D A102/B98, GT105: moving mean100→102 là có ích, nhưng moving mean tiếp tới110 vẫn tăng log-ratio A/B dù xa GT hơn. Tối ưu preference tương đối không bảo đảm tối ưu vị trí tuyệt đối. Cần native anchor, bước cập nhật kiểm soát và audit box thực tế; làm mới/giảm ưu tiên cặp đã sửa là biến thể sau.

Với covariance cố định:

\[
\nabla_\mu(\log q(u^+)-\log q(u^-))=\Sigma_C^{-1}(u^+-u^-).
\]

Đây là hướng ở cấp mean đầu ra. Jacobian về tham số, các mẫu khác, native loss, clipping và AdamW có thể thay tác động thực tế. Không suy từ công thức này rằng IoU chắc chắn tăng. Sigma và beta cùng ảnh hưởng thang margin; phải audit/tune có ngân sách, không chuyển nguyên lambda cũ rồi tuyên bố strength tương đương.

Trước train geometry: test các tình huống lệch tâm/quá rộng/quá hẹp/đã tốt/nhiều kích thước, kiểm normalization/density/gradient/finite difference, rồi kiểm bước AdamW trên ảnh thật. Geometry experiment phải có trainable mask cho regression và đối chứng tương ứng; không trộn vào V0.

## 11. Đánh giá cơ chế, confidence và tổng quát hóa

### Phân rã box/score bằng checkpoint sẵn có

| Boxes | Scores/classes | Vai trò |
|---|---|---|
| S0 | S0 | Mốc |
| S0 | DPO | Diagnostic classification/ranking |
| DPO | S0 | Diagnostic geometry |
| DPO | DPO | Model thật |

Cùng raw slot layout, preprocessing, class space, batch precision, NMS/cap. Không trộn các detections đã NMS một cách tùy tiện; recombine raw-aligned outputs rồi chạy cùng postprocessing. Các delta không cộng tuyến tính do argmax/NMS. Đây không phải model mới hoặc phép tách nhân quả đầy đủ.

So thêm S0 chọn threshold trên validation để phát hiện lợi ích chỉ do điểm vận hành. Không gọi mọi thay đổi confidence là calibration: calibration là mức phù hợp giữa score và xác suất đúng, cần phép đo riêng. mAP/đườngPR trên full low-threshold predictions, recall tạiFP budget tương đương và ordinary confidence0,25 bổ sung cho nhau. Nếu threshold tuned, khóa trước test và báo tách khỏi protocol chính.

### Metric tối thiểu

- Micro/macro recall, FP/ảnh, mAP50-95, AP/recall/support theo class.
- GT recovered/lost/persistent-miss theo cùng ảnh/ID/matcher; net recovery=recovered-lost tương ứng giảmFN tổng.
- Score/class gap/ordinary-IoU ở GT mục tiêu và GT S0 đã đúng. Fixed-slot diagnostics không thay recall trên mọi candidate.
- IoU trên GT cảhai model cùng phát hiện kèm coverage/recovered/lost, tránh survivor bias.
- Density/overlap/object-size strata; pool coverage và repeat thực tế của từng class.
- DPO/native norms, cosine, clip, độ lệch score/box và paired AdamW checks. Norm% không bằng% tác động lên update.

### Split và kiểm chứng

- Train: sinh cặp, native/DPO, gradient audit. Không dùng val/test GT để chọn ảnh/slot cần sửa.
- Validation: development/checkpoint/threshold lựa chọn. Validation hiện đã được dùng nhiều lần để suy luận thiết kế, nên không nhận là bằng chứng test độc lập.
- Train probe bỏ khỏi DPO nhưng native dùng vẫn là train; bỏ khỏi cảfine-tune thì cũng đã được S0 học, chỉ là holdout của giai đoạn tiếp tục train.
- Test: không dùng tạo cặp/gradient/tuning; đánh giá sau khi khóa phương pháp. Nếu xem test rồi sửa thiết kế, không tiếp tục gọi kết quả test đó là kiểm định cuối chưa bị ảnh hưởng.
- Test ngoài nguồn chụp, cùng không gian class, cần thiết cho tuyên bố chuyển miền; test cùng bộ không đủ chứng minh mọi điều kiện ngoài thực tế. Nếu chưa có dữ liệu đó thì giới hạn tuyên bố, không tự tạo độ tin cậy.
- Kiểm ảnh nguồn/augment liên quan giữa splits. Khi bootstrap, lấy đơn vị ảnh hoặc nhóm ảnh nguồn, không coi từng lá cùng ảnh là mẫu độc lập. Vòng nhiều seed dùng cặp so có kiểm soát; CI/SD không tự thay cho kiểm chứng nguồn ảnh mới.

Augmentation, refresh cặp và unfreeze backbone là các thí nghiệm sau. Nếu augment, cùng input/transform/GT cho A/B, regenerate candidates và giữ source-GT IDs qua transform; không so hai ảnh biến đổi khác nhau như hai đáp án.

## 12. Đối chứng và thứ tự công việc

| Bước | Công việc | Điều được phép kết luận |
|---|---|---|
| A | Phân rã raw boxes/scores trên checkpoint cũ; audit pair pool | Xác định thành phần thay đổi/độ bao phủ; không khẳng định causal benefit mới |
| B | Validate policy giữ/bỏ, immutable C, loss/replay/one-step | Cơ chế có được triển khai đúng không |
| C | Pilot matched native-only vs native+DPO, cùngmask/stream/budget | Đóng góp thêm DPO trong protocol V0 |
| D | So supervised local-BCE trên cùng candidates, cùngtuning budget | Giá trị thêm của reference/preference so hard-example supervision |
| E | Nhiều seed khi có tín hiệu; class/group/geometry ablations từng biến | Tính lặp lại và đóng góp thành phần |
| F | Khóa cấu hình rồi test, ngoài nguồn nếu có | Kết quả độc lập trong đúng phạm vi dữ liệu đã đo |

Native-only và local-BCE là đối chứng thí nghiệm, không là hướng nghiên cứu KB2 riêng. Giữ tổng images/native/DPO exposure, sốupdate, optimizer, trainable mask và tuning budget để giải thích chênh lệch. So với S0 luôn báo, nhưng không lấy riêng native+DPO>S0 làm bằng chứng lợi ích riêng DPO.

Trước pilot phải khóa primary metric, FP budget, mAP tolerance, checkpoint selection và gate đi tiếp. Mục tiêu tối thiểu: ordinary validation có net recovery tốt hơn đối chứng, class-level không bị đánh đổi không giải thích được, FP trong budget; không chỉ margin/GT train tăng. Chưa có ngưỡng gain tối thiểu và FP/mAP tolerance được người dùng chốt; đây là tham số thiết kế còn mở, không đổi hậu nghiệm sau khi xem kết quả.

## 13. Artifact và việc cần chốt trước code

Mỗi pair cần: objective/policy version, source/policy/reference hash, image/source IDs, transform, slot layout, GT ID nếu có, decision type/class, context ID/hash, hai lựa chọn, rendered outputs, matcher/NMS/cap/threshold protocol, reason chosen, per-GT transitions/FP change, sampler metadata. Context dùng chung có thể lưu một lần/ảnh-view, cặp lưu phần khác biệt để giảm I/O.

Replay chỉ tính xác suất các action đã lưu với cùng conditioning. Cache generation log-prob chỉ là audit; reference live cùng batch dùng trong loss. Cặp/policy mới phải có version riêng, không trộn manifest checkpoint whole-action lịch sử và nhận là cùng objective.

Các điểm còn cần khóa hoặc đo:

1. Ngưỡng candidate/negative ambiguity, matching/tie-breaking, sốcandidate/ảnh, coverage theo density/class.
2. Candidate đã có detection khác cùng GT: cặp keep/drop có thực sự đổi outcome không? Không gán rescue chỉ vì thêm duplicate.
3. Renderer cho selected candidate dưới0,25 và stable NMS priority; kiểm immutable C trước/sauNMS đúng phạm vi.
4. Trainable mask V0, mixed native stream, sampler/repeat/tỷlệpreservation, cách cân bằng λ và tuning budget.
5. GT-target score không được tăng nhờ chấm lại action có box khác. Kiểm raw-box invariance ởV0 và geometry drift khi mởregression.
6. Cách xây bounded candidate groups; treatment của crop/border/degenerate boxes, no-valid-pair, class không có mẫu, unlabeled/ambiguous negatives mà không mở phạm vi sửa annotation.
7. Model selection/FP budget/mAP tolerance, calibration control, sốseed/CI, external-test availability.
8. Nếu refresh, reference vẫnS0; context/proposal priorities và vòng generation được version hóa, không âm thầm làm preferences đổi trong replay.

## 14. Đánh giá khả thi sau review

| Thành phần | Đánh giá |
|---|---|
| Audit dữ liệu/cặp và phép phân rã | Khả thi với đầu ra raw hiện có; chưa chạy phân rã mới |
| V0 giữ/bỏ có đối chứng | Khả thi về mặt kỹ thuật, có thể kiểm tra bằng công thức/gradient/replay |
| Giảm FN mà không tăng FP/không mất lá khác | Giả thuyết thực nghiệm chưa được chứng minh |
| DPO tốt hơn supervised hard examples | Chưa có bằng chứng; cần đối chứngD |
| NMS groups/geometry Gaussian | Hợp lý để nghiên cứu, chưa là cấu hình được chọn hoặc kiểm chứng |
| Tổng quát hóa sang nguồn ảnh mới | Chưa xác nhận, không thể suy từ train margin hoặc audit |

Con số30–40% từng nêu trong trao đổi chỉ là ước lượng chủ quan, không được hiệu chỉnh từ dữ liệu; không dùng như xác suất khoa học để chọn thiết kế hay tăng ngân sách.

## 15. Nguồn tham chiếu

Các bài sau hỗ trợ khái niệm nền; không chứng minh các mở rộng KB2 ở tài liệu này sẽ hiệu quả:

- DPO gốc tối ưu preference bằng log-ratios với reference và loss dạng classification: [Rafailov và cộng sự](https://arxiv.org/html/2305.18290v3).
- Confidence calibration khác với việc chỉ đổi score/threshold; temperature scaling được nghiên cứu trong classification, không tự chứng minh chuyển sang detector này: [Guo và cộng sự](https://arxiv.org/abs/1706.04599).
- NMS có thể loại object chồng lấn; giữ NMS cố định trong thí nghiệm DPO trước khi đổi postprocessing: [Bodla và cộng sự](https://arxiv.org/abs/1704.04503).

Đây là thiết kế để chuyển thành các thí nghiệm kiểm chứng nhỏ. Không có code train mới, checkpoint mới hay kết quả Gaussian/local-DPO mới được tạo trong lần tổng hợp này.

## 16. Cập nhật thực nghiệm: tách native background, 31/10/2026

Sau gradient audit, đã thử đúng một thay đổi: giữ native foreground, box/DFL,
assignment và normalization; chỉ nhân0,5 vào BCE ở các slot background do native
assigner xác định. Không giảm target0 ở các class khác tại foreground slot.
Implementation tách trong `conditional_bg/`, giữ fingerprint của conditional V0.

Đã chạy GPU đủ bốn cấu hình background1/0,5 × native-only/native+DPO,2.000
update mỗi cấu hình, seed42, cùng bank/input stream; full479 validation images.
Tại confidence0,25, DPO gốc FN911/FP820→DPO mới FN872/FP959; S0 FN866/FP984.
GT mất so S0 giảm49→15, GT cứu4→9. Native-only mới cùng FN872, FP958;
DPO chưa tạo khác biệt GT identities ở threshold này. Bốn best vẫn S0.

Tại cùng FP984 với cutoff quét trên validation, native-only mới FN864, DPO mới
FN862, S0 FN866. Đây là tín hiệu nhỏ exploratory, chưa chứng minh lợi ích ổn
định hoặc tổng quát hóa. Không tự đổi threshold chính/default hoặc chạy dài.

51 tests đạt, gồm GPU kiểm alpha1 loss/gradient trùng bản cũ và alpha0,5 chỉ
giảm gradient background. Fresh checkpoint validation khớp logs, checkpoint cũ
giữ nguyên. Chi tiết: [review native background](report_final/KB2_NATIVE_BACKGROUND_PILOT_20261031.md).

Theo yêu cầu tiếp theo, đã khởi chạy lượt xác nhận dài cùng background0,5:
DPO trước và native-only sau,15.000 bước mỗi lượt, khởi tạo lại S0, không đổi
các tham số còn lại. **Đã hoàn tất và fresh GPU validation khớp log:** DPO15k
FN886/FP881, native-only FN886/FP878, S0 FN866/FP984. Từ2k lên15k, DPO cứu
GT9→31 nhưng mất GT15→51, FN872→886; cảhai best vẫn S0.

Tại cùng FP984, DPO FN859 và native-only FN860. Cảhai cứu cùng44 GT S0 bỏ sót;
DPO giữ thêm1 GT vốn S0 đã đúng. Tín hiệu riêng DPO rất nhỏ, chưa đủ xác nhận
lợi ích ổn định. Không chỉ tăng step tiếp hoặc tự thay loss sau review này.
Xem [review dài](report_final/KB2_NATIVE_BACKGROUND_LONG_REVIEW_20261031.md),
[plan và log](report_final/KB2_NATIVE_BACKGROUND_LONG_RUN_20261031.md).

## 17. Cập nhật01/11: pilot sức mạnh DPO và audit box ranking

Theo mục tiêu cải thiện nhiều metric, đã chạy bốn λ0/0,01/0,03/0,1, mỗi nhánh2k,
giữ background0,5/beta0,1/LR/mask/bank/input stream. Selector mới lưu mọi mốc
0/1k/2k, best_AP không có recall gate, kèm Pareto mAP/P/R. Cảbốn best ở1k.

λ0,1/1k mAP38,0617%, P51,2601%, R53,8951%, FP967/FN870; native1k mAP38,0625%,
P51,2355%, R53,8421%, FP967/FN871. So S0 mAP37,8218%/P50,9227%/R54,1070%/
FP984/FN866, có cải thiện nhiều metric kèm giảm recall nhẹ. Lợi thế riêng DPO
chưa rõ: cùngFP984 tại1k, λ0,1 FN865 so native864. Không tự kéo dài hoặc tăngλ tiếp.

Audit20 cặp candidate-swap hợp lệ từ2.411 ảnh train,15 train/5 probe. Xét hết
proposal với nguyên gate chỉ tăng28 cặp/22 train trên20 ảnh train.237 lần
thử bị loại ở config gốc vì candidate tốt vẫn bị NMS suppress sau khi bỏ một
candidate kém. Chưa train nhánh mới bằng lượng cặp ít này, không nới gate hậu nghiệm. Bước thiết
kế tiếp cần xử lý cụm candidate cạnh tranh và sự nhất quán giữa renderer/loss/
ordinary prediction; không chỉ thay hai box trong một context vẫn chặn box tốt.

Xem [pilot λ](report_final/KB2_DPO_STRENGTH_PILOT_20261101.md) và
[candidate audit](report_final/KB2_CANDIDATE_RANK_AUDIT_20261101.md).
