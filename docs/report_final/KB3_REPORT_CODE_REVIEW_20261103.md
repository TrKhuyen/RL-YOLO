# Review báo cáo KB3 so với code và artifacts — 03/11/2026

Đối tượng: [Report_KB3_15_10_2026.md](Report_KB3_15_10_2026.md). Review tĩnh source/config, đối chiếu snapshot 15/10 và kết quả hiện lưu; không train, không chạy inference mới, không sửa code hoặc báo cáo gốc. Không kiểm chứng lại từng paper/link học thuật.

**Kết luận:** phần phương pháp A–TPE và quality B/PPO nhìn chung đúng với implementation. Báo cáo là snapshot lịch sử hợp lệ, nhưng thiếu kết quả pretrained hoàn tất nếu dùng để trình bày trạng thái hiện tại. Có một phát biểu về guard không đúng cho mọi stage và một số giới hạn bảo đảm kỹ thuật cần ghi rõ hơn. Chưa thấy bằng chứng từ review này rằng các scores đã báo bị tính sai.

## 1. Phát hiện cần xử lý

### F1 — Ưu tiên cao nếu dùng làm báo cáo kết quả hiện tại: pretrained đã hoàn tất 8/8

Vị trí báo cáo: mục 1.1, 13, 15.2 và kết luận tại dòng 1774. Câu “2/8 completed, chưa có winner” **đúng tại snapshot**, không phải lỗi lịch sử: `report_kb3_20261015_evidence.json` xác nhận timestamp `2026-10-15T02:31:17.4204133+07:00`, trial 2 đang chạy.

Tuy nhiên, [evaluation.json](../../kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n/evaluation.json) hiện ghi đủ tám trial; `status.json` ghi evaluate/complete. Các số dưới đây được đọc từ artifacts, không được suy từ ngày sửa file:

| Trial | Mode | Canonical mAP50–95 (%) | Δ KB1 lịch sử (điểm %) |
|---:|---|---:|---:|
| 0 | Anchor | 36,0677 | −1,7530 |
| 1 | Startup random | 37,7225 | −0,0982 |
| 2 | TPE | **38,5071** | **+0,6864** |
| 3 | TPE | 37,0877 | −0,7330 |
| 4 | TPE | 38,3459 | +0,5252 |
| 5 | TPE | 36,2870 | −1,5337 |
| 6 | TPE | 37,2655 | −0,5552 |
| 7 | TPE | 38,2293 | +0,4086 |

Ranking: **2 → 4 → 7 → 1 → 6 → 3 → 5 → 0**. Tổng 1.387 epoch, 22,7038 giờ theo `result.elapsed_seconds`; `test_evaluated=false`.

Winner 2: `lr0=0.005354944281299181`, `weight_decay=0.0009130337123429847`, momentum 0,937, augmentation abstraction 0,5; seed 0; native-selected epoch 139; train thực 169 epoch. Cải thiện so anchor mới **+2,4394 điểm %**. Cả tám selected checkpoint còn tồn tại trong cây thư mục hiện tại, SHA256 khớp `result.json`; tám result ghi cùng một initialization hash.

**Bổ sung nên làm:** giữ nguyên snapshot 15/10, thêm phụ lục “kết quả sau snapshot” hoặc một báo cáo cập nhật riêng. Kết luận hợp lệ là “ứng viên TPE vượt KB1 lịch sử trên validation trong một search seed”; chưa đủ kết luận TPE vượt random search, cải thiện ổn định nhiều seed, cải thiện test hoặc thành công của RL. Chênh lệch anchor–KB1 vẫn cần giữ trong phần giới hạn.

### F2 — Ưu tiên vừa: guard giảm target chỉ áp dụng trong search

Vị trí báo cáo: mục 11.3, dòng 740: “Giảm target thấp hơn completed work → Từ chối để không che kết quả đã có”.

Trong [tpe_pipeline.py](../../kb3_hyperparameter_optimization/quality/tpe_pipeline.py), kiểm tra trial đã hoàn tất vượt target chỉ nằm trong `TPEExperiment.search()` (dòng 207–212). `evaluate()` gọi `records()`; `records()` chỉ đọc `range(self.target)` (dòng 174–185), không kiểm tra các trial nằm ngoài target. `write_evaluation()` ghi đè báo cáo ở root bằng số record được truyền vào.

Vì thế, trong môi trường gốc thỏa manifest, `--stage evaluate --trials 4 --resume` trên output có tám trial có thể tạo lại root report bốn trial; guard trong search không chặn đường này. Báo cáo versioned tám trial vẫn còn, weights không bị xóa, nhưng root report không còn phản ánh toàn bộ completed work. Đây là kết luận từ control flow; review không chạy lệnh ghi đè để thử.

**Cách sửa nội dung:** ghi guard hiện có chỉ áp dụng stage search/all. Nếu muốn bảo đảm chung như báo cáo mô tả, cần đưa kiểm tra target vào đường dùng chung cho evaluate và search; đó là sửa code riêng.

### F3 — Ưu tiên vừa: thiếu giới hạn tái lập khi di chuyển Windows → Linux

Vị trí báo cáo: mục 7.5, 11.2–11.3 và 22.3. Báo cáo đã nói artifacts bị gitignore, nhưng chưa mô tả đầy đủ phụ thuộc đường dẫn tuyệt đối.

Các result hiện lưu `best_checkpoint` và `run_dir` dạng `D:\graduate\RL-YOLO\...`. `validated_result()` dùng trực tiếp `Path(result['best_checkpoint'])` để tìm file. `trajectory_diagnostics()` cũng đọc `Path(result['run_dir']) / 'epochs.json'`. Trên workspace Linux hiện tại, các path đó không trỏ tới selected checkpoint cục bộ dù file tương ứng vẫn có trong thư mục trial. Vì vậy copy artifacts sang Linux chưa đủ để chạy lại CLI evaluate/resume nguyên trạng; diagnostic còn có thể không tìm được curves.

Ngoài ra fingerprint chứa path tuyệt đối trong settings và relative path theo biểu diễn hệ điều hành. Sau khi chuẩn hóa dấu phân cách chỉ để đối chiếu hash thủ công, **toàn bộ source KB3 được manifest liệt kê vẫn khớp bytes hiện tại**, nhưng sáu source KB1 khác hash:

- `adapters/yolov5_adapter.py`
- `dataloader.py`
- `evaluate.py`
- `run_provenance.py`
- `run_verified.py`
- `train_supervised.py`

Danh sách trên thuộc `kb1_reward_guided_training/`. Khác byte không tự chứng minh khác hành vi; nhưng resume guard so hash sẽ không coi đây là source gốc. Không gọi kết quả artifacts là sai vì source hiện đã thay đổi; cũng không khẳng định CLI hiện có thể tái lập run cũ nguyên trạng.

**Bổ sung nên làm:** ghi rõ môi trường/đường dẫn gốc và yêu cầu source snapshot tương ứng; nếu xây công cụ đọc artifacts sau di chuyển, cần resolve đường dẫn có kiểm SHA và giữ provenance gốc. Không sửa manifest/result gốc để vượt guard.

### F4 — Ưu tiên thấp: phân biệt kiểm tra cấu trúc với chứng thực nguồn COCO

Vị trí báo cáo: mục 3.4, dòng 149, diễn đạt checkpoint “có 80 lớp COCO”. `verify_pretrained_source()` trong [pretrained_reference.py](../../kb3_hyperparameter_optimization/quality/pretrained_reference.py) kiểm SHA so với source vừa resolve, checkpoint có `nc=80`, và tên/shape tensors khớp kiến trúc. Kiểm tra này không tự chứng minh model đã được pretrained trên COCO hoặc giống official weights: một checkpoint ngẫu nhiên đúng layout cũng có thể thỏa các điều kiện cấu trúc đó.

Báo cáo đã có bằng chứng audit storages official riêng ở mục 13.6; nên dẫn bằng chứng này khi khẳng định nguồn COCO, và mô tả runtime guard là kiểm tra integrity/layout/transfer. Đây là chỉnh độ chính xác của lời mô tả, không phải phát hiện nguồn YOLOv8n hiện tại bị sai.

### F5 — Ưu tiên thấp: fingerprint và resume không xác thực lại mọi bằng chứng

Vị trí báo cáo: mục 11.2. Fingerprint thực sự hash source KB1/KB3, dataset và các settings/dependency versions đã liệt kê; mô tả đó đúng. Nhưng nó không hash bytes source Ultralytics hoặc toàn bộ môi trường Python/CUDA/driver. Cùng chuỗi version không bảo đảm package không bị sửa tại chỗ.

`validated_result()` kiểm checkpoint SHA, parameters/seed, selector/init, cờ `pretrained_backbone_verified`, và miền hợp lệ của primary score. Nó không chạy lại tensor-transfer proof hoặc canonical evaluation khi resume; không hash lại `epochs.json`/`initial.json` để chứng thực toàn bộ trajectory. `search()` tái dựng proposal/history, còn `records()` dùng cho evaluate không tái dựng proposal như search.

**Bổ sung nên làm:** phân biệt “guard hợp đồng và integrity checkpoint” với “tái kiểm chứng độc lập mọi metric/trajectory”. Hiện không có bằng chứng tampering; đây là giới hạn của cơ chế kiểm tra.

## 2. Những nội dung đã đối chiếu và thấy phù hợp

| Nội dung báo cáo | Đối chiếu source/config |
|---|---|
| Default A-only; TPE không phải RL | Hai shell scripts gọi `quality.tpe_pipeline`; config `rl_enabled=false` |
| Fine-tune độc lập từ source KB1 args, best KB1 chỉ làm reference | `resolve_pretrained_recipe()`, `prepare_reference()`, pretrained trainer |
| Search LR/WD log scale; momentum/augmentation giữ nguyên | `local_distributions()`, `propose()`, trainer guards |
| Anchor 0, random 1, TPE từ 2; sampler seed 42+i | `n_startup_trials=2`, enqueue anchor, dựng study từ completed history |
| Native selection/patience; canonical chỉ chấm best/last | `pretrained_detector.py`; callback final_eval được loại khỏi history |
| Native fitness mAP50–95, lịch linear/warmup, WD batch scaling, EMA | Source Ultralytics 8.4.163 hiện lưu trong `.venv/Lib/site-packages/ultralytics/` |
| Canonical P/R/F1 operating và AR300 khác nhau | `canonical_eval.py`, `_canonical_metrics()` |
| B còn scratch, liên tục một trainer; 20 observation dimensions, masks bốn heads | `quality/detector.py`, `state.py`, `_ActorCritic` |
| Reward best-gain ×100, gamma=1; GAE lambda 0,95 | `best_gain()`, `QualityPPO.update()` |
| PPO clip 0,2, value 0,5, entropy 0,01, KL 0,02 | `quality/ppo.py` |
| Class rollout 256/64 nhưng quality config 64/32 | `QualityPPO` defaults, `kb3_quality.yaml` |
| Không mid-trial exact resume | Trial dở archive rồi train lại; completed result được tái sử dụng |

Kiểm tra bổ sung: các Markdown file links cục bộ trong báo cáo đều resolve tới file tồn tại. Điều này không đồng nghĩa external URLs hoặc mọi anchor đã được kiểm tra.

## 3. Phân tích còn thiếu để hoàn thiện kết quả hiện tại

Các mục này phần lớn đã được báo cáo nêu là công việc tương lai; không nên trình bày như lỗi phương pháp đã bị bỏ qua hoàn toàn:

1. Bảng đủ tám pretrained trial, native/canonical best/last, hyperparameters và tổng chi phí; cập nhật kết luận dựa trên trial 2.
2. Ví dụ thực về selector mismatch: trial 3 có canonical last **37,6367%**, cao hơn native-selected best **37,0877%** khoảng **0,5490 điểm %**. Báo cáo đã nêu giới hạn lý thuyết; dữ liệu mới minh họa rõ. Không tự thay selected best bằng last sau khi xem scores.
3. Matched multi-seed confirmation của anchor/winner; hiện tám trial chỉ có một detector seed. Chưa có đối chứng random search với ngân sách tương đương và nhiều search seeds.
4. Giải thích hoặc giới hạn rõ chênh anchor mới 36,0677% với historical KB1 37,8207%. Tăng so anchor không đồng nghĩa toàn bộ tăng đó do TPE so với pipeline KB1 lịch sử.
5. Support theo class/size và per-class diagnostics trước khi kết luận AP small; kết quả test sau freeze, kèm test-access history đã được báo cáo khai báo.
6. Nếu luận văn cần bằng chứng RL: thực nghiệm B/PPO trên recipe phù hợp, frozen-policy evaluation và đối chứng tương ứng. Tám trial TPE hoàn tất vẫn không thay thế bằng chứng đó.

## 4. Phạm vi kiểm chứng

Đã đọc source/config các đường A/B liên quan, source runtime Ultralytics được lưu, snapshot cũ, evaluation/result hiện tại; tự tính lại deltas và tổng hợp kết quả; kiểm SHA256 thực của tám selected checkpoint; so source hashes với manifest sau chuẩn hóa path cho việc đọc. Không chạy test suite lịch sử, không rerun detector hoặc đo lại metrics/latency. Các số pass của suite cũ trong báo cáo vẫn là bằng chứng từ logs lịch sử, không phải kết quả test mới của review này.
