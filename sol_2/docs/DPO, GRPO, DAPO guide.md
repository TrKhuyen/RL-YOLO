# Tóm tắt DPO, GRPO, RLHF và các biến thể liên quan

Tài liệu này tổng hợp lại các video YouTube về RLHF, DPO, GRPO và các phương pháp tinh chỉnh mô hình ngôn ngữ lớn. 

## RLHF Explained

### Tổng quan về quá trình huấn luyện AI

- Pre-training: xây dựng nền tảng từ lượng dữ liệu rất lớn.
- Supervised Fine-tuning (SFT): tinh chỉnh có giám sát để biến mô hình thành dạng hội thoại.
- Optimization: bước cuối dùng ít dữ liệu hơn nhưng chất lượng cao hơn để tối ưu hiệu suất.

### RLHF

#### Khái niệm học tăng cường

- Huấn luyện mô hình bằng quá trình thử và sai trong một môi trường động.
- Ví dụ con chuột trong mê cung: phô mai đóng vai trò phần thưởng, giúp chuột học cách đi đến đích.

#### Cách thức hoạt động của RLHF

- Policy: mô hình cuối cùng mà ta muốn điều khiển hành vi.
- Reward Model: mô hình riêng để chấm điểm hoặc xếp hạng câu trả lời.
- PPO (Proximal Policy Optimization): thuật toán phổ biến để cập nhật mô hình dựa trên điểm thưởng.
- KL Divergence: cơ chế phanh để mô hình không đi quá xa hoặc gian lận.

#### Thách thức của RLHF

- Khó ổn định vì có nhiều thành phần phụ thuộc lẫn nhau.
- Tốn kém vì vẫn cần con người dán nhãn dữ liệu sở thích ban đầu.
- Có sự mâu thuẫn giữa các người gắn nhãn khác nhau.

### DPO

#### Bước đột phá: bỏ mô hình phần thưởng

- DPO coi mô hình ngôn ngữ như một mô hình phần thưởng ngầm.
- Thay vì huấn luyện reward model riêng, DPO tối ưu trực tiếp trên dữ liệu sở thích.
- Dữ liệu thường gồm ba cột: Prompt, Chosen và Rejected.

#### Ưu điểm của DPO

- Đơn giản hơn vì không cần reward model riêng.
- Nhanh hơn vì bỏ được bước thử và sai phức tạp.
- Ổn định hơn và chịu được temperature cao hơn mà không sụt chất lượng đột ngột.

### KTO

#### Cảm hứng từ kinh tế học hành vi

- KTO lấy ý tưởng từ Kahneman và Tversky, dựa trên Prospect Theory.
- Trọng tâm là con người nhạy với mất mát hơn là lợi ích tương đương.

#### Sự khác biệt về dữ liệu

- KTO không cần so sánh cặp tốt hơn - tệ hơn như DPO.
- Chỉ cần dữ liệu thích - không thích.
- Hoạt động tốt ngay cả khi dữ liệu bị lệch.

#### Ưu điểm vượt trội của KTO

- Có thể biến mô hình thô thành mô hình hội thoại mà không cần SFT.
- Khử nhiễu tốt với dữ liệu mâu thuẫn hoặc không rõ ràng.
- Có các tham số để điều chỉnh mức sáng tạo hoặc bảo thủ.

### Kết luận

- RLHF là framework truyền thống, mạnh nhưng phức tạp.
- DPO là lựa chọn thay thế nhanh, đơn giản và ổn định hơn cho nhiều bài toán.
- KTO phù hợp với các nguồn dữ liệu sở thích kiểu like/dislike.

## Direct Preference Optimization: Your Language Model is Secretly a Reward Model | DPO paper explained

### 1. Giới thiệu về tinh chỉnh mô hình và DPO

- LLM ban đầu học dự đoán từ tiếp theo từ dữ liệu internet.
- Tuy nhiên mô hình không phải lúc nào cũng trả lời theo cách con người mong muốn.
- Mục tiêu của tinh chỉnh là tạo ra phản hồi hữu ích, hội thoại tốt và an toàn hơn.

### 2. RLHF truyền thống

#### Quy trình bốn bước

- Bắt đầu với LLM đã được huấn luyện sơ bộ.
- Con người chọn câu trả lời tốt hơn trong các cặp câu trả lời.
- Huấn luyện reward model để học cách chấm điểm.
- Dùng RL để điều chỉnh mô hình gốc sao cho tạo ra câu trả lời có điểm cao hơn.

#### Nhược điểm

- Dễ gặp reward hacking.
- Tốn điện toán và dễ mất ổn định khi huấn luyện.

### 3. DPO

#### DPO hoạt động như thế nào

- DPO loại bỏ reward model và RL vòng ngoài.
- Mô hình học trực tiếp từ dữ liệu ưu tiên của con người.
- Loss được xây dựng để tăng xác suất câu trả lời được thích và giảm xác suất câu trả lời bị loại.

#### Vì sao trước đây dùng RLHF thay vì DPO

- Thói quen kỹ thuật khiến RLHF trở thành mặc định.
- Nhiều người từng nghĩ reward model là cần thiết để mở rộng quy mô dữ liệu.
- Thực tế sau này cho thấy dữ liệu dán nhãn của con người đã đủ lớn và hữu ích.

### 4. Hiệu quả thực tế và thử nghiệm

- DPO được thử trên các mô hình như GPT-J và Pythia.
- Kết quả được đánh giá bằng GPT-4 thay cho con người.
- DPO cho win rate tốt hơn so với RLHF truyền thống trong thí nghiệm này.
- Cộng đồng mã nguồn mở đã áp dụng DPO cho các mô hình lớn hơn như Llama-2 và Zephyr.

### 5. Kết luận

- DPO nhanh hơn vì không cần reward model riêng.
- DPO ổn định hơn vì tránh được sự phức tạp của RL.
- DPO dùng các loss quen thuộc trong deep learning để đạt mục tiêu căn chỉnh mô hình.

## Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math

### 1. Giới thiệu về mô hình ngôn ngữ và căn chỉnh AI

- LLM là mô hình xác suất dự đoán token tiếp theo.
- Quá trình sinh văn bản là lặp dự đoán token, đưa token mới vào prompt và tiếp tục.
- Căn chỉnh AI nhằm dạy mô hình cư xử theo cách con người mong muốn.

### 2. Ôn tập học tăng cường

#### Agent, state, action, reward

- Agent: tác nhân thực hiện hành động.
- State: trạng thái hiện tại.
- Action: hành động được chọn.
- Reward: điểm số đánh giá kết quả.
- Policy: quy tắc chọn hành động.

#### Liên hệ với mô hình ngôn ngữ

- Prompt đóng vai trò trạng thái.
- Việc chọn token tiếp theo đóng vai trò hành động.
- Ta muốn tối ưu policy để tạo ra câu trả lời nhận phần thưởng cao hơn.

### 3. Từ RLHF đến mô hình Bradley-Terry

#### Vấn đề của việc chấm điểm trực tiếp

- Con người thường khó cho điểm tuyệt đối.
- Con người giỏi hơn khi so sánh hai câu trả lời với nhau.

#### Mô hình Bradley-Terry

- Mô hình này chuyển dữ liệu so sánh thành xác suất và phần thưởng.
- Xác suất chọn winner cao hơn loser phụ thuộc vào chênh lệch reward.

### 4. Giải mã DPO

#### Hạn chế của RLHF truyền thống

- RLHF cần reward model riêng và thuật toán như PPO.
- Cách này phức tạp và tốn tài nguyên.

#### Ý tưởng đột phá của DPO

- DPO chứng minh không cần reward model riêng.
- Bằng cách dùng hiệu số giữa winner và loser, DPO có thể triệt tiêu hằng số chuẩn hóa khó tính.
- Bài toán RL được đổi thành một loss function trực tiếp.

#### Mục tiêu kép của DPO

- Tối đa hóa reward cho câu trả lời tốt hơn.
- Giữ mô hình không thay đổi quá xa so với mô hình gốc bằng KL constraint.

### 5. Triển khai thực tế và log probabilities

#### Cách tính log probabilities

- Đưa prompt và answer vào Transformer.
- Tính logits ở từng vị trí token.
- Dùng log-softmax để lấy log probability của token đúng.
- Cộng lại để có xác suất của cả câu.

#### Sử dụng Hugging Face

- Có thể huấn luyện bằng DPOTrainer.
- Cần mô hình huấn luyện, mô hình tham chiếu và dữ liệu winner/loser.
- Tham số beta điều chỉnh mức độ mô hình được phép thay đổi.

### Kết luận

- DPO loại bỏ nhu cầu dùng RL trong việc căn chỉnh mô hình.
- Cách này nhanh hơn, ổn định hơn và ít tốn tài nguyên hơn RLHF truyền thống.

## Fine-tuning LLMs on Human Feedback (RLHF + DPO)

### 1. Động lực: tại sao phải tinh chỉnh mô hình?

- Base model mạnh nhưng chưa chắc trả lời theo cách hữu ích.
- Prompt engineering có thể giúp, nhưng rất tốn công và không ổn định.
- Fine-tuning giúp mô hình phục vụ người dùng tốt hơn.

### 2. Instruct GPT

- Instruct GPT được tạo ra để giải quyết vấn đề alignment.
- GPT-3 gốc chỉ dự đoán token tiếp theo trên internet.
- Instruct GPT được dạy để đưa ra các bước thực hiện cụ thể và hữu ích.

### 3. Quy trình huấn luyện ba bước của OpenAI

- Pre-training: tạo mô hình gốc.
- SFT: dạy mô hình cách đối thoại bằng dữ liệu mẫu.
- RLHF: bước quan trọng giúp mô hình vượt trội hơn SFT đơn thuần.

### 4. RLHF

#### Cách thức hoạt động

- RLHF giúp mô hình học bằng thử và sai.
- Reward model thay con người trong vòng lặp huấn luyện.

#### Huấn luyện reward model bằng xếp hạng

- Với một prompt, mô hình sinh nhiều câu trả lời.
- Con người xếp hạng chúng từ tốt đến tệ.
- Reward model học cách định lượng chất lượng từ các so sánh này.

#### PPO

- PPO là thuật toán cập nhật mô hình dựa trên điểm số reward.
- Nó giữ sự thay đổi trong mỗi bước ở mức an toàn để quá trình huấn luyện ổn định.

### 5. DPO

#### Hạn chế của RLHF

- Cần nhiều mô hình cùng lúc.
- Phụ thuộc vào chất lượng reward model.

#### Sự đơn giản của DPO

- DPO biến bài toán thành supervised learning trên cặp thắng - thua.
- Không cần reward model riêng biệt.

### 6. Ví dụ thực tế: tinh chỉnh tiêu đề YouTube

#### Bài toán

- Tinh chỉnh Qwen 2.5 để viết tiêu đề YouTube theo gu cá nhân.
- Mục tiêu là tránh kiểu tiêu đề quá máy móc và sáo rỗng.

#### Quy trình thực hiện

- Viết ý tưởng video.
- Dùng AI tạo nhiều tiêu đề cho mỗi ý tưởng.
- Tự tay chọn tiêu đề tốt hơn trong các cặp so sánh.
- Huấn luyện bằng TRL và DPOTrainer.

#### Kết quả đánh giá

- Mô hình DPO được ưu tiên hơn trong phần lớn trường hợp.
- Các tiêu đề sau DPO trực diện và đúng ý hơn.

### Kết luận

- RLHF là tiêu chuẩn vàng nhưng phức tạp.
- DPO là giải pháp thực dụng để cá nhân hóa mô hình theo gu riêng.
- Chất lượng dữ liệu ưu tiên vẫn là yếu tố quyết định.

## Group Relative Policy Optimization (GRPO) Visualized

### 1. Giới thiệu về PPO truyền thống

- PPO dùng old policy để sinh token tiếp theo.
- Advantage cho biết một hành động tốt hơn hay tệ hơn mong đợi.
- PPO truyền thống thường cần reward model và value model.

#### Kiểm soát sự thay đổi

- Clip giúp cập nhật an toàn hơn.
- KL divergence giữ mô hình mới không đi quá xa mô hình gốc.

### 2. Sự ra đời của DeepSeek R1 và vấn đề với reward model

- Trong các bài toán tư duy, reward model dễ bị lạm dụng.
- DeepSeek dùng rule-based reward thay vì chấm điểm bằng mô hình học.
- Các quy tắc cứng như kiểm tra đáp án toán, test code, hoặc định dạng suy nghĩ giúp giảm reward hacking.

### 3. GRPO

#### Cách hoạt động

- Mô hình sinh một group nhiều câu trả lời cho cùng một prompt.
- Mỗi câu trả lời được chấm điểm reward.
- Advantage được tính bằng cách so với trung bình của cả nhóm.

#### Tại sao gọi là Group Relative

- Mô hình chỉ cần tốt hơn mức trung bình của chính nhóm nó sinh ra.
- Không cần biết thế nào là hoàn hảo tuyệt đối.

### 4. Tổng kết công thức và ứng dụng

- Policy ratio so sánh mô hình mới và cũ.
- Group advantage là điểm so với trung bình nhóm.
- Clipped objective giúp cập nhật ổn định.
- KL penalty giữ mô hình không mất gốc.

### 5. DeepSeek R1-Zero và R1

- R1-Zero đi thẳng từ base model sang GRPO mà không cần SFT.
- R1 bắt đầu bằng dữ liệu chain of thought chọn lọc rồi mới dùng GRPO.

## GRPO Reinforcement Learning Explained (DeepSeekMath Paper)

### 1. Giới thiệu về kỷ nguyên mô hình suy luận

- Sau o1, các mô hình suy luận trở thành tâm điểm.
- Test-time scaling cho phép mô hình dành thêm thời gian để suy nghĩ trước khi trả lời.
- DeepSeek R1 là cột mốc lớn với GRPO là lõi huấn luyện.

### 2. Quy trình huấn luyện DeepSeek Math

#### Lựa chọn base model

- DeepSeek dùng DeepSeek-Coder-Base-v1.5 7B làm nền tảng.
- Bắt đầu từ mô hình giỏi code giúp học toán tốt hơn.

#### Math pre-training

- Dùng quy trình lặp để lọc dữ liệu toán chất lượng cao.
- Tạo ra hàng chục triệu trang web toán học sạch hơn.
- Mô hình kết quả là DeepSeek-Math-Base 7B.

#### Instruction tuning

- Mô hình được dạy trả lời theo yêu cầu với bộ dữ liệu bài toán và lời giải chi tiết.
- Kết quả là DeepSeek-Math-Instruct 7B.

### 3. GRPO là gì?

- GRPO là một thuật toán RL cải tiến từ PPO.
- Điểm nổi bật là loại bỏ value model để tiết kiệm tài nguyên.

### 4. Cải tiến của GRPO

- GRPO yêu cầu mô hình tạo một nhóm câu trả lời cho cùng một câu hỏi.
- Lợi thế được tính bằng reward trừ đi reward trung bình của nhóm.
- Cách làm này giảm đáng kể chi phí bộ nhớ và độ phức tạp.

### 5. Chi tiết kỹ thuật

- Policy ratio so sánh xác suất trước và sau khi cập nhật.
- Clipping giữ cập nhật nhỏ và ổn định.
- Dù reward ở mức câu trả lời, tín hiệu vẫn truyền xuống từng token.

### 6. Hai cơ chế giám sát

- Outcome supervision: chỉ chấm đúng/sai ở kết quả cuối.
- Process supervision: chấm từng bước suy luận nhỏ.

### 7. Kết luận

- DeepSeek-Math 7B cho thấy tối ưu thuật toán có thể bù cho hạn chế tài nguyên.
- GRPO là nền tảng quan trọng chứng minh hiệu quả của huấn luyện theo nhóm.

## How does GRPO work?

### 1. Khái niệm cơ bản về GRPO và so sánh với các phương pháp cũ

- SFT học từ ground truth, đẩy xác suất của câu đúng lên.
- ORPO dùng cả chosen và rejected để học trực tiếp trên cặp ưu tiên.

### 2. Cách GRPO hoạt động theo nhóm

- Mô hình sinh nhiều câu trả lời trong một group.
- Reward của từng câu được so với baseline của các câu còn lại.
- Câu nào tốt hơn trung bình được khuyến khích; câu nào kém hơn bị giảm ưu tiên.

### 3. Hệ thống phần thưởng trong GRPO

#### Đo lường phần thưởng

- Accuracy reward kiểm tra kết quả đúng hay sai.
- Format reward kiểm tra mô hình có tuân thủ cấu trúc mong muốn hay không.

#### Vấn đề của lấy mẫu ngẫu nhiên

- Nếu cả nhóm đều tệ, mô hình có thể không nhận được tín hiệu học hữu ích.
- Do đó chất lượng khởi đầu và cách lấy mẫu vẫn rất quan trọng.

### 4. Lịch sử và sự tiến hóa của các thuật toán

- TRPO giới hạn cập nhật trong vùng tin cậy nhưng rất phức tạp.
- PPO đơn giản hóa TRPO bằng clipping.
- GRPO tiếp tục đơn giản hóa bằng cách bỏ value model.

### 5. Kết luận

- Phép màu của GRPO nằm ở verified rewards và cách so sánh nhóm.
- Dữ liệu đúng/sai rõ ràng vẫn là lõi của quá trình học.

## LLM Training & Reinforcement Learning from Google Engineer | SFT + RLHF | PPO vs GRPO vs DPO

### 1. Tổng quan về quy trình huấn luyện LLM hiện đại

- Pre-training dùng dữ liệu không gắn nhãn để học cấu trúc ngôn ngữ và kiến thức chung.
- Post-training gồm SFT và RLHF để làm mô hình hữu ích và an toàn hơn.

### 2. Supervised Fine-tuning (SFT)

- SFT dùng tập dữ liệu nhỏ nhưng chất lượng cao do con người gắn nhãn.
- Giúp mô hình thích nghi với nhiệm vụ cụ thể như tóm tắt, chăm sóc khách hàng hoặc giải toán.

### 3. Reinforcement Learning (RL)

- RL giúp mô hình học bằng thử và sai.
- Agent, environment, action, reward và policy là các khái niệm cốt lõi.
- Trong ví dụ Flappy Bird, chim là agent và điểm số là reward.

### 4. Các thuật toán tối ưu hóa chính

#### PPO

- PPO là thuật toán tiêu chuẩn được dùng rộng rãi cho ChatGPT.
- Nó cần nhiều mô hình chạy song song nên khá phức tạp.

#### GRPO

- GRPO loại bỏ value model và dùng nhóm câu trả lời để so sánh.
- Cách này tiết kiệm tài nguyên và giảm độ phức tạp hệ thống.

#### DPO

- DPO bỏ qua bước huấn luyện reward model.
- Chỉ cần so sánh trực tiếp giữa câu trả lời tốt hơn và tệ hơn.

### 5. Vì sao cần RL trong post-training?

- Dữ liệu SFT có giới hạn.
- RL giúp mô hình thích nghi với nhiệm vụ suy luận logic và tương tác phức tạp.
- RL cũng dạy mô hình những điều không nên làm.

### 6. Ví dụ thực tế: quy trình từ đầu đến cuối

- Bước 1: dùng SFT để dạy mô hình trả lời chuẩn.
- Bước 2: tạo reward model từ các xếp hạng của con người.
- Bước 3: dùng PPO hoặc phương pháp tương tự để tối ưu mô hình với KL divergence giữ ổn định.

### Kết luận chung

- Huấn luyện LLM hiện đại đã chuyển từ chỉ dựa vào pre-training sang post-training thông minh hơn.
- GRPO và DPO đang làm cho quá trình này rẻ hơn, đơn giản hơn và hiệu quả hơn PPO truyền thống.
