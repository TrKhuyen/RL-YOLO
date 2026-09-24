# KB3 — Review logic hyperparameter optimization

## 1. Ranh giới hai kịch bản

### KB3-A: Traditional HPO

Mỗi trial chọn **một cấu hình tuyệt đối** trước khi train, khởi tạo detector từ cùng một nguồn pretrained và giữ cấu hình đó trong toàn run. Validation chọn trial/config và checkpoint; sau search, khóa best config rồi train lại trên seed đánh giá chưa dùng trong search. [Fixed trial runner](../../kb3_hyperparameter_optimization/traditional_hpo/runner.py#L46), [Random Search](../../kb3_hyperparameter_optimization/traditional_hpo/random_search.py) và [TPE](../../kb3_hyperparameter_optimization/traditional_hpo/optuna_search.py) thể hiện đúng logic này.

### KB3-B: Adaptive RL

Một episode huấn luyện **một detector** qua các segment. Sau mỗi segment, agent nhận state từ loss/validation/progress/hyperparameter, chọn action thay đổi hyperparameter cho segment tiếp theo; reward phản ánh thay đổi chất lượng và các penalty đã định nghĩa. [Environment](../../kb3_hyperparameter_optimization/envs/yolo_hpo_env.py#L50) và [search space](../../kb3_hyperparameter_optimization/search_space.py) đúng với ý tưởng này. PPO dùng state; bandit không dùng state nên là đối chứng action learning, không phải policy điều khiển theo trạng thái.

Sự khác nhau về **phương pháp** đã được tách đúng. Điều kiện chưa đạt chắc chắn là sự liên tục của state train trong backend thực và protocol so sánh hai kịch bản.

## 2. Luồng triển khai hiện có

| Thành phần | Code | Đánh giá logic |
|---|---|---|
| Fixed trial | [traditional_hpo/runner.py](../../kb3_hyperparameter_optimization/traditional_hpo/runner.py#L46) | Một vector h được truyền vào một lời gọi train toàn run. Đúng với HPO truyền thống. |
| Adaptive state/action/reward | [env](../../kb3_hyperparameter_optimization/envs/yolo_hpo_env.py#L90), [reward](../../kb3_hyperparameter_optimization/reward.py#L22) | State, action và reward được tách rõ. Action là thay đổi tương đối so với hyperparameter hiện tại, có clipping. |
| Policy | [PPO](../../kb3_hyperparameter_optimization/agents/ppo_agent.py), [bandit](../../kb3_hyperparameter_optimization/agents/bandit_agent.py), [evaluate_policy.py](../../kb3_hyperparameter_optimization/evaluate_policy.py) | Có policy training, checkpoint và evaluation frozen. Cần bảo đảm seed evaluation không dùng trong search/policy training. |
| Backend thực | [ultralytics_worker.py](../../kb3_hyperparameter_optimization/adapters/ultralytics_worker.py#L126) | Chạy Ultralytics theo segment, lưu last checkpoint và resume. Có callback áp hyperparameter. Tính continuity phải được kiểm tra ở cấp state. |
| Backend mô phỏng | [cli.py](../../kb3_hyperparameter_optimization/cli.py), [simulated.py](../../kb3_hyperparameter_optimization/adapters/simulated.py) | Có ích cho unit/smoke test, không chứng minh optimizer, EMA, scheduler và data order của Ultralytics thật. |
| Tổng hợp | [compare_hpo.py](../../kb3_hyperparameter_optimization/compare_hpo.py) | Chất lượng lấy từ best metrics và thời gian từ final metrics là đúng ý tưởng; comparator chưa kiểm tra tính tương thích của input. |

## 3. Kiểm tra điều kiện continuity của KB3-B

Yêu cầu nghiên cứu là trong một episode, **model, optimizer, scheduler và EMA phải liên tục** khi agent đổi hyperparameter. Worker hiện dùng nhiều lần gọi train: tại segment sau, [nạp last checkpoint với resume](../../kb3_hyperparameter_optimization/adapters/ultralytics_worker.py#L145). Trong phiên bản Ultralytics đang cài trong môi trường review, checkpoint train lưu EMA, optimizer và scaler nhưng không lưu raw model riêng và không lưu scheduler state. Khi resume, raw model được tái tạo từ EMA và scheduler được tạo lại theo cấu hình/epoch. Process mới cũng tạo trainer/dataloader và khởi tạo RNG lại.

Điều này cho thấy:

- **Optimizer và EMA** có cơ chế khôi phục, nhưng vẫn cần kiểm tra state trước/sau boundary.
- **Raw model** không chắc bằng raw weights ngay trước khi dừng segment; dùng EMA làm điểm tiếp tục là một thay đổi trạng thái.
- **Scheduler** được tái dựng, không duy trì object/state nguyên vẹn. Nếu scheduler chỉ là hàm theo epoch, kết quả *có thể* tương đương trong một số trường hợp, nhưng không được khẳng định nếu chưa so learning-rate trace.
- **Data order/augmentation RNG** có thể thay đổi tại boundary do tạo trainer/dataloader mới.

Vì vậy câu “một episode giữ liên tục model, optimizer, scheduler, EMA” là **mục tiêu thiết kế**, chưa là thuộc tính được chứng minh của backend thực. Cách tốt nhất là giữ một trainer sống trong một process và đổi hyperparameter qua callback ở ranh giới segment. Nếu tiếp tục dùng resume, cần gọi phương pháp chính xác là *segmented resume from EMA checkpoint* và thêm fixed-schedule control cũng dùng cùng boundary để tách ảnh hưởng của policy khỏi ảnh hưởng của resume.

Integration test quyết định: chạy một lịch hyperparameter cố định theo hai đường, train liên tục và train chia segment; tại mỗi boundary so raw weights, EMA, optimizer buffers, scaler, scheduler/LR, sample order và output model. Test này phải chạy trên backend thực; unit test mô phỏng không thay thế được.

## 4. Lỗi logic và điều kiện cần sửa

| ID | Ưu tiên | Phát hiện | Sửa/tiêu chí đóng |
|---|---|---|---|
| K3-L1 | Cao | Continuity giữa segment chưa đạt hoặc chưa chứng minh như mục 3. So KB3-A một call với KB3-B nhiều resume dễ lẫn ảnh hưởng của boundary. | Integration equivalence; sửa sang trainer liên tục hoặc thêm đối chứng fixed segmented với đúng cùng cơ chế resume. |
| K3-L2 | Cao | [Worker](../../kb3_hyperparameter_optimization/adapters/ultralytics_worker.py#L231) lấy best checkpoint theo metric native rồi mới chấm canonical checkpoint đó. Nếu ranking canonical khác native, canonical-best có thể chưa được chọn. | Chọn và báo cáo cùng một evaluator/metric; lưu đúng cặp checkpoint–metric, hoặc công bố rõ native selection và canonical chỉ là phép đánh giá ngoài. |
| K3-L3 | Cao | Khi không dùng canonical evaluator, [worker](../../kb3_hyperparameter_optimization/adapters/ultralytics_worker.py#L76) điền AP-small bằng placeholder trong khi [reward](../../kb3_hyperparameter_optimization/reward.py#L38) vẫn dùng AP-small. Reward không phản ánh thành phần mong muốn. | Nếu reward cần AP-small, buộc backend cung cấp metric thật; thiếu metric thì fail fast, không thay bằng hằng số. |
| K3-L4 | Cao | [Comparator](../../kb3_hyperparameter_optimization/compare_hpo.py#L13) không kiểm tra cùng model, split, evaluator, seed, budget và phase. Nó có thể so search runs với frozen-evaluation runs. | Schema kết quả phải ghi các trường này; comparator chỉ nhận kết quả tương thích và seed ghép cặp; tách chi phí search khỏi chất lượng policy/config đã khóa. |
| K3-L5 | Vừa | KB3-A lấy cấu hình tuyệt đối trong bounds, KB3-B đi qua action tương đối rời rạc từ initial. Hai nhánh cùng bounds nhưng không có cùng tập cấu hình có thể đạt. | Báo cáo reachable action space và mức clipping; giới hạn diễn giải so sánh theo đúng chiến lược được khảo sát, thêm đối chứng phù hợp nếu muốn so tối ưu thuần. |
| K3-L6 | Vừa | [Callback worker](../../kb3_hyperparameter_optimization/adapters/ultralytics_worker.py#L159) đổi args/optimizer ở các thời điểm khác nhau; scheduler và warmup có thể ghi đè LR/momentum dự kiến. | Log requested và applied hyperparameter theo param group/epoch; test action thay đổi chỉ segment kế tiếp và tạo tác dụng đúng thành phần. |
| K3-L7 | Vừa | Time penalty trong [reward](../../kb3_hyperparameter_optimization/reward.py#L44) chỉ có tác dụng khi time budget được thiết lập. Cấu hình mặc định có thể khiến term này không hoạt động. | Hoặc định nghĩa time budget trước khi train policy, hoặc bỏ term và mô tả reward không tối ưu thời gian; cost vẫn báo cáo riêng. |
| K3-L8 | Vừa | Overfit penalty dùng loss train/val tổng hợp; scale có thể thay theo model/hyperparameter, nên không tự động phản ánh overfitting. | Chuẩn hóa hoặc kiểm tra sensitivity/ablation reward; dùng cùng định nghĩa qua A/B, không đưa test vào reward. |
| K3-L9 | Vừa | Search A và policy training B dùng seed schedule khác nhau; cùng tổng epoch chưa đồng nghĩa ngân sách tìm kiếm và mức nhiễu giống nhau. | Công bố seed schedule, số detector train, GPU time, số validation, lỗi và early stop; đánh giá hai phương pháp trên cùng seed mới. |

## 5. Protocol logic sau khi sửa

1. Khóa model/pretrained, split dữ liệu, search bounds, evaluator và metric chính. Không chọn phương pháp dựa trên test.
2. KB3-A: mỗi trial chọn h tuyệt đối, train từ cùng nguồn, h cố định toàn run; chỉ validation chọn config.
3. KB3-B: agent học policy qua episode; mỗi episode giữ đúng state train theo định nghĩa đã kiểm chứng; random schedule và fixed segmented là đối chứng quan trọng.
4. Khóa best config A và policy B trước evaluation. Đánh giá trên cùng seed chưa dùng trong search, cùng checkpoint-selection metric và cùng evaluator.
5. Báo cáo chất lượng của **best checkpoint** và chi phí của **toàn quá trình**; ghi riêng search cost và final evaluation cost. Chỉ sau khi chạy lại mới kết luận A hay B tốt hơn.

Báo cáo này không điền kết quả hoặc bộ hyperparameter tối ưu vì dữ liệu đã được làm lại và thí nghiệm chưa chạy lại.

