'''Summarize only completed, verified KB1 runs and orchestrate seed42 screening.'''
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results' / 'screening_seed42_v2'
PREFLIGHT = ROOT / 'results' / 'screening_preflight_v2'
REPORT = ROOT / 'docs' / 'KB1_SCREENING_REPORT.md'
MODELS = ['yolov5s', 'yolov8n', 'yolov8s', 'yolov11n', 'yolov11s', 'dp_yolo']
MODES = ['native_only', 'kb1b']
METRICS = ['mAP50', 'mAP50_95', 'AR300', 'precision', 'operating_recall', 'f1', 'macro_f1']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, deltas, statuses = [], [], []
    for model in MODELS:
        runs = {}
        for mode in MODES:
            folder = OUTPUT / model / mode
            if (folder / 'completed.json').exists():
                runs[mode] = read(folder / 'completed.json')
                statuses.append((model, mode, 'complete', runs[mode]['steps_run']))
            elif (folder / 'status.json').exists():
                status = read(folder / 'status.json')
                statuses.append((model, mode, status['status'], status.get('step', 0)))
            else:
                statuses.append((model, mode, 'pending', 0))
        if runs:
            initial = next(iter(runs.values()))['baseline']
            rows.append(dict(Model=model, Stage='supervised', seed=42, checkpoint_step=0, **initial))
        for mode, run in runs.items():
            rows.append(dict(Model=model, Stage=mode, seed=42,
                             checkpoint_step=run['best_step'], **run['best']))
        if len(runs) == 2:
            a, b, n = runs['kb1b']['baseline'], runs['kb1b']['best'], runs['native_only']['best']
            for key in METRICS:
                if abs(a[key] - runs['native_only']['baseline'][key]) > 1e-6:
                    raise RuntimeError('Baseline mismatch for ' + model)
            deltas.append(dict(Model=model, **{
                key + suffix: b[key] - ref[key]
                for suffix, ref in [('_kb1b_minus_supervised', a), ('_kb1b_minus_native_only', n)]
                for key in METRICS}))
    for name, data in [('results_full.csv', rows), ('results_delta.csv', deltas)]:
        if data:
            path = OUTPUT / name
            tmp = path.with_suffix('.tmp')
            with tmp.open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
            tmp.replace(path)
    completed = sum(s[2] == 'complete' for s in statuses)
    audit_path = ROOT / 'results/dataset_audit/summary.json'
    audit = read(audit_path) if audit_path.exists() else None
    lines = [
        '# BÁO CÁO KB1 — KIỂM CHỨNG VÀ SCREENING SEED 42',
        '',
        'Cập nhật: ' + datetime.now().isoformat(timespec='seconds') + '.',
        f'**Trạng thái: {completed}/12 lượt train hoàn tất; {len(deltas)}/6 cặp đủ kết quả so sánh.**',
        '',
        '## 1. Phạm vi và tên gọi',
        '',
        'KB1 là Huấn luyện YOLO hai giai đoạn: baseline có giám sát và tinh chỉnh có hướng dẫn bởi reward. '
        'KB1-A tạo best checkpoint riêng cho sáu model. KB1-B tiếp tục tối ưu từ chính checkpoint '
        'tương ứng, rồi so sánh trước/sau trong từng model. Nhánh native-only là đối chứng tiếp tục '
        'train, dùng để đánh giá phần bổ sung reward và proxy trong KB1-B.',
        '',
        'Tên phương pháp chính xác: reward-guided fine-tuning (RL-inspired). '
        'Prediction xác định sau NMS và log-confidence là surrogate; không phải DPO hay REINFORCE chính xác.',
        '',
        '## 2. Các vấn đề đã phát hiện và sửa',
        '',
        '- Dataloader cũ âm thầm resize trực tiếp khi augmentation lỗi. Nhãn vượt biên rất nhỏ '
        'do làm tròn tọa độ đã thực sự kích hoạt lỗi này. Bản mới clip góc box trước transform, '
        'kiểm tra ảnh/nhãn và dừng rõ ràng nếu còn lỗi.',
        '- Cố định seed augmentation theo seed/epoch/chỉ số ảnh, cùng thứ tự batch giữa hai nhánh. '
        'Sampler có thể tiếp tục đúng vị trí sau khi ngắt.',
        '- Giữ cố định trọng số tích phân DFL của YOLOv8/11; đóng băng backbone/neck và thống kê BatchNorm.',
        '- Native-only trực tiếp tính native loss + L2-SP. Không tạo prediction giả hoặc ghi reward giả.',
        '- Best checkpoint được cập nhật theo score cao nhất. Min-delta chỉ điều khiển early stopping; '
        'checkpoint tăng nhẹ vẫn được giữ.',
        '- Lưu optimizer, EMA, RNG và bước train để resume tại ranh giới accumulation; '
        'hash code/checkpoint ngăn resume khác cấu hình. Ghi file qua file tạm rồi thay thế.',
        '- Tính matching box trên CPU rồi lấy score có gradient trên GPU để giảm đồng bộ trong vòng lặp.',
        '',
        'Audit một epoch augmentation đã đọc đủ **2.072 ảnh train**, thu được **7.929 box** sau augmentation. '
        'Đây là kiểm tra kỹ thuật dữ liệu, không phải chỉ số chất lượng model.',
        '',
        '## 3. Evaluator chung',
        '',
        'Protocol: **kb1_canonical_v2_ar300**. Chọn checkpoint và screening trên **592 ảnh validation**. '
        'Tập test chưa dùng để tuning.',
        '',
        '| Thành phần | Quy tắc |',
        '|---|---|',
        '| Input | RGB float32 [0,1]; resize cạnh dài 640; pad giữa bằng 114 |',
        '| Nhãn | Clip góc box trước transform; xyxy trong ảnh 640×640 |',
        '| AP | Confidence ≥0,001; IoU 0,50:0,05:0,95 |',
        '| NMS | Theo lớp; IoU 0,60; tối đa 300 box/ảnh |',
        '| Chọn lớp | Một lớp có score cao nhất trên mỗi candidate |',
        '| Metric | torchmetrics với faster-coco-eval; cap300 |',
        '| AR300 | Average Recall qua các ngưỡng IoU, tối đa 300 detection |',
        '| P/R/F1 | Confidence 0,25; matching cùng lớp; IoU 0,50 |',
        '| APs/APm/APl | Diện tích box trong ảnh đã resize, không phải ảnh gốc |',
        '',
        'AR300 khác operating_recall. COCO cap100 cũng là protocol hợp lệ; ở đây cố định cap300 '
        'và ghi tên riêng. Không trộn điểm evaluator cũ với điểm mới. Ví dụ YOLOv8n baseline '
        'mAP50–95 đổi từ khoảng 0,4603 sang 0,4739 sau sửa tiền xử lý; đây không phải tăng do train.',
        '',
        '## 4. Phương pháp train và đối chứng',
        '',
        'Hai nhánh cùng seed 42, cùng checkpoint KB1-A, cùng augmentation, chỉ train Detect head. '
        'Adam, batch 8, accumulation 2 = 16 ảnh/update. LR 5e-7; warmup 500 micro-step từ 1e-7, '
        'sau đó cosine về 1e-7. Ngân sách tối đa 10.000 micro-step = 5.000 optimizer update, '
        'eval mỗi 500 bước. Early stopping sau 4 lần eval không tăng quá 0,0005 so với mốc patience. '
        'Cùng ngân sách tối đa nhưng số bước thực tế có thể khác nhau do early stopping.',
        '',
        '**Native-only:** L = 0,25 L_native + 0,001 L2-SP.',
        '',
        '**KB1-B:** L = 0,25 L_native + 0,001 L2-SP + 0,10 L_reward + 0,25 L_proxy.',
        '',
        'YOLOv5/DP-YOLO dùng ComputeLoss và hyp gắn trong checkpoint (box/objectness/classification). '
        'YOLOv8/11 dùng native criterion Ultralytics (box/classification/DFL). Loss chia batch size '
        'để quy về mỗi ảnh. Thang native loss khác nhau theo họ model; cùng hệ số không đồng nghĩa '
        'cùng độ lớn gradient. Đối chứng chính là ghép cặp trong từng model.',
        '',
        'YOLOv5/DP-YOLO load qua backend fuse Conv/BN với feature extractor đóng băng. '
        'YOLOv8/11 giữ model chưa fuse, BN eval và DFL cố định. Đây là continued fine-tuning bằng '
        'native criterion trong cấu hình chung, không tái hiện đầy đủ native trainer.',
        '',
        'Reward gộp precision/recall/IoU theo trọng số 0,25/0,45/0,30 trên IoU 0,5/0,6/0,7/0,8. '
        'EMA chuẩn hóa advantage. Proxy tăng confidence TP, giảm FP và dùng max candidate score '
        'cho FN. FN proxy chưa định vị riêng từng GT bị bỏ sót; reward không truyền gradient qua box/NMS. '
        'B−native đo đóng góp kết hợp reward và proxy, chưa tách riêng reward.',
        '',
        'DP-YOLO screening cố định W3F=OFF, PSA=OFF theo mặc định code hiện tại; native loss dùng '
        'CIoU và assignment chuẩn. Chưa có metadata môi trường KB1-A chứng minh baseline đã bật '
        'W3F/PSA. Tên model không đủ để khẳng định hai thành phần này đã tham gia huấn luyện.',
        '',
        'Score chọn best = 0,4 mAP50 + 0,4 mAP50–95 + 0,2 AR300. Step 0 cũng có thể là best '
        'sau một lượt đã hoàn tất. Chỉ completed.json xác nhận lượt hoàn tất.',
        '',
        '## 5. Kết quả validation',
        '',
        'Chỉ đưa vào bảng các lượt hoàn tất và đã kiểm tra best checkpoint nạp lại. '
        'Chỉ số hiển thị theo %, delta là điểm phần trăm.',
        '',
        '| Model | Giai đoạn | Best step | mAP50 | mAP50–95 | AR300 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |'.format(
            r['Model'], r['Stage'], r['checkpoint_step'],
            r['mAP50']*100, r['mAP50_95']*100, r['AR300']*100))
    lines += ['', '| Model | ΔmAP50–95 B−A | ΔmAP50–95 B−native |',
              '|---|---:|---:|']
    for d in deltas:
        lines.append('| {} | {:+.3f} | {:+.3f} |'.format(d['Model'],
            d['mAP50_95_kb1b_minus_supervised']*100,
            d['mAP50_95_kb1b_minus_native_only']*100))
    lines += ['', '## 6. Đánh giá và hướng tiếp theo', '']
    if len(deltas) == 6:
        wins = sum(d['mAP50_95_kb1b_minus_native_only'] > 0 for d in deltas)
        mean = sum(d['mAP50_95_kb1b_minus_native_only'] for d in deltas)/6*100
        lines.append(f'KB1-B vượt native-only về mAP50–95 trên {wins}/6 model. '
                     f'Delta trung bình không trọng số là {mean:+.3f} điểm phần trăm.')
        lines.append('Đây là một seed trên validation đã dùng chọn best, chưa chứng minh ý nghĩa '
                     'thống kê hay hiệu quả test. Cần phân tích gradient native/proxy/reward, '
                     'ablation native+proxy và native+proxy+reward, rồi thử hệ số loss hoặc mở neck. '
                     'Sau khi chọn cấu hình bằng validation, lặp seed 43/44 và đánh giá test cuối.')
    else:
        lines.append('Đang chạy screening; chưa đủ sáu cặp để kết luận tổng quát hoặc chọn tối ưu tiếp.')
    lines += ['', '## 7. Trạng thái và bằng chứng', '',
              '| Model | Nhánh | Trạng thái | Bước đã ghi nhận |', '|---|---|---|---:|']
    lines += ['| {} | {} | {} | {} |'.format(*s) for s in statuses]
    lines += ['', '- results_full.csv: điểm từng model/giai đoạn; results_delta.csv: B−A và B−native.',
              '- Mỗi nhánh: manifest.json, baseline.json, train.jsonl, validation.jsonl, '
              'best.pt, last.pt và completed.json.',
              '- Smoke test hai bước chỉ kiểm chứng kỹ thuật, không dùng làm kết quả nghiên cứu.',
              '- Kiểm tra tự động: nạp checkpoint vào model dựng mới; gradient hữu hạn/nonzero; '
              'head thay đổi; frozen parameters và BN không đổi; best checkpoint khớp metric sau reload.',
              '- Test hành vi: NMS/gradient, matching duplicate/sai lớp, hướng gradient proxy, '
              'letterbox nhãn sát biên, sampler/augmentation sau resume và evaluator prediction hoàn hảo.',
              '- Kết quả: ../results/screening_seed42_v2/. Lõi train: run_verified.py.',
              '']
    lines += ['## 8. Kết quả smoke test của runner hiện tại', '',
              '| Model | native-only | KB1-B |', '|---|---|---|']
    for model in MODELS:
        values = []
        for mode in MODES:
            path = PREFLIGHT / model / mode / 'completed.json'
            values.append('PASS (2 bước)' if path.exists() else 'Chưa hoàn tất')
        lines.append('| {} | {} | {} |'.format(model, *values))
    lines += ['', 'Bộ test hành vi: 6/6 đạt; bộ kiểm tra KB1 có sẵn: 19/19 đạt. '
              'Các test kiểm tra code, không xác nhận tập validation/test độc lập với train.', '']
    if audit and audit['status'] == 'leakage_detected':
        lines[4:4] = [
            '',
            '## Kết luận ưu tiên: tạm dừng screening dài vì rò rỉ dữ liệu',
            '',
            '**149/592 ảnh validation (25,17%) và 67/296 ảnh test (22,64%) có cùng ID nguồn với train.** '
            'Đây là ảnh gốc và các bản flip nằm ở các tập khác nhau. Hai cặp được kiểm chứng bằng pixel: '
            'sau lật đúng hướng, tương quan lần lượt 0,999855 và 0,999483; sai khác trung bình '
            '0,578 và 1,548 trên thang 0–255, phù hợp với lưu lại JPEG.',
            '',
            'Train–validation trùng 115 nhóm nguồn; train–test trùng 60 nhóm; validation–test '
            'trùng 38 nhóm. Kiểm tra dựa trên ID nguồn đầy đủ, không chỉ trùng tên ngắn. '
            'Có thể còn dạng trùng khác mà phép kiểm tra này chưa bao phủ.',
            '',
            'Code pre-data/split_dataset.py shuffle từng ảnh rồi cắt tập, không group theo nguồn. '
            'Code augment_class.py tạo hậu tố _augN_hflip/vflip/hvflip. Nếu chia sau khi gộp '
            'ảnh augmentation, thuật toán này cho phép rò rỉ đã quan sát; chưa có log để xác nhận '
            'chính xác lệnh lịch sử tạo dataset hiện tại.',
            '',
            '### Dấu hiệu dùng để kết luận có leak',
            '',
            '1. **Cùng ID nguồn xuất hiện ở nhiều split.** Sau khi chỉ bỏ hậu tố augmentation '
            'có quy tắc `_augN_hflip`, `_augN_vflip` hoặc `_augN_hvflip`, phần tên còn lại '
            'giống hoàn toàn, bao gồm cả mã Roboflow dài. Ví dụ ảnh train '
            '`000gb_jpg.rf.eeb741072e585dc54b4332616793266f.jpg` đi cùng ảnh validation '
            '`000gb_jpg.rf.eeb741072e585dc54b4332616793266f_aug2_hvflip.jpg`.',
            '',
            '2. **Hậu tố mô tả đúng phép biến đổi trong code.** `augment_class.py` tạo ảnh bằng '
            '`flip_image_and_boxes`, rồi đặt tên `_aug{n}_{flip_str}`. Đây là quan hệ sinh ảnh '
            'có chủ đích, không phải hai ảnh độc lập tình cờ có tên gần giống.',
            '',
            '3. **Nội dung pixel xác nhận quan hệ đó.** Với hai cặp mẫu, lật ảnh train theo đúng '
            'hậu tố làm tương quan pixel với ảnh validation đạt 0,999855 và 0,999483. MAE chỉ '
            '0,578 và 1,548 trên thang 0–255; phần sai khác nhỏ phù hợp với nén JPEG lại.',
            '',
            '4. **Cách chia tập tạo điều kiện cho hiện tượng này.** `split_dataset.py` shuffle '
            'từng file ảnh riêng lẻ rồi cắt danh sách thành train/valid/test. Script không nhóm '
            'ảnh gốc với các biến thể augmentation theo source ID trước khi chia.',
            '',
            '### Leak ảnh hưởng như thế nào?',
            '',
            '| Thành phần | Ảnh hưởng |',
            '|---|---|',
            '| Validation | Model được chọn checkpoint trên các mẫu có nội dung đã xuất hiện trong train; '
            'mAP, AR, precision, recall và F1 có nguy cơ lạc quan hơn dữ liệu hoàn toàn mới. |',
            '| Early stopping và best checkpoint | Validation dễ hơn có thể làm thay đổi best step và '
            'quyết định dừng, nên ảnh hưởng trực tiếp tới checkpoint cuối của cả KB1-A và KB1-B. |',
            '| Test | 67/296 ảnh test cùng nguồn với train, nên test hiện tại không còn là phép đánh giá '
            'độc lập đáng tin cậy về khả năng tổng quát hóa. |',
            '| So sánh model | Mọi model dùng cùng split nên bảng vẫn có giá trị chẩn đoán nội bộ, nhưng '
            'mức thiên lệch có thể khác theo kiến trúc và khả năng bất biến với flip; thứ hạng có thể đổi '
            'trên split sạch. |',
            '| Delta KB1-B | B−A và B−native-only vẫn hữu ích như tín hiệu kỹ thuật trong cùng protocol, '
            'nhưng KB1-B chọn checkpoint trên validation bị leak nên chưa thể coi delta là bằng chứng '
            'tổng quát hóa. |',
            '| Độ chắc chắn thống kê | Các bản flip cùng nguồn là các mẫu tương quan mạnh, làm số quan sát '
            'độc lập thực tế nhỏ hơn số file; khoảng tin cậy nếu coi mọi file độc lập sẽ quá lạc quan. |',
            '',
            'Audit chứng minh sự giao nhau theo source ID và xác minh pixel trên hai cặp đại diện. '
            'Nó không định lượng chính xác số điểm mAP bị tăng do leak; muốn biết mức thiên lệch phải '
            'đánh giá lại trên split sạch. Phép kiểm tra hiện tại cũng có thể bỏ sót ảnh trùng nội dung '
            'nhưng mang ID khác, vì vậy các con số trên là bằng chứng tối thiểu đã xác nhận.',
            '',
            'Vì vậy, các metric hiện có vẫn mô tả tập đánh giá này nhưng không đủ để chứng minh '
            'khả năng tổng quát hóa. Smoke test đạt không có nghĩa thiết kế thực nghiệm đã hợp lệ. '
            'Việc train dài được chặn trước khi khởi động; chưa hoàn tất screening hoặc tối ưu hiệu năng.',
            '',
            '**Phương án đề xuất:** tạo dataset mới ở thư mục riêng, chia nhóm theo ảnh nguồn trước '
            'augmentation; chỉ augment train, giữ validation/test là ảnh gốc; audit không giao nguồn '
            'và kiểm tra phân bố đủ lớp. Sau đó chạy lại KB1-A cho sáu model và hai nhánh KB1-B seed 42. '
            'Audit tìm thấy 2.463 ID nguồn đầy đủ và cả 2.463 nhóm đều còn ảnh gốc, nên có thể '
            'chuẩn bị split mới từ ảnh gốc; vẫn cần kiểm tra các nguồn trùng nội dung nhưng khác ID. '
            'Checkpoint hiện tại đã học trên split cũ, nên chia lại tập rồi tiếp tục dùng chúng '
            'không tự loại bỏ rò rỉ. KB1-A mới cần khởi tạo lại từ pretrained chung ban đầu. '
            'Cần thống nhất việc mở rộng sang chạy lại KB1-A trước khi thực hiện.',
            '',
            'Bằng chứng: ../results/dataset_audit/summary.json và overlap_pairs.csv. '
            'Dataset và checkpoint gốc chưa bị thay đổi. Kiểm tra test ở đây chỉ audit nguồn ảnh, '
            'không đánh giá model hoặc dùng metric test để tuning.',
            '',
        ]
    tmp = REPORT.with_suffix('.tmp')
    tmp.write_text('\n'.join(lines), encoding='utf-8')
    tmp.replace(REPORT)
    return completed


def main():
    if '--report-only' in sys.argv:
        summarize()
        return
    from audit_splits import audit
    if audit()['status'] == 'leakage_detected':
        summarize()
        raise RuntimeError('Dataset source leakage detected; see KB1_SCREENING_REPORT.md before training')
    for model in MODELS:
        for mode in MODES:
            if not (PREFLIGHT / model / mode / 'completed.json').exists():
                raise RuntimeError(f'Missing preflight: {model}/{mode}')
    summarize()
    for model in MODELS:
        for mode in MODES:
            folder = OUTPUT / model / mode
            folder.mkdir(parents=True, exist_ok=True)
            print('START', model, mode, flush=True)
            command = [sys.executable, '-u', str(ROOT / 'run_verified.py'), '--model', model,
                       '--mode', mode, '--output', str(OUTPUT)]
            with (folder / 'console.log').open('a', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=ROOT.parent, stdout=log, stderr=subprocess.STDOUT)
            summarize()
            if result.returncode:
                raise RuntimeError(f'Run failed: {model}/{mode}; see console.log')
            print('DONE', model, mode, flush=True)
    print('SCREENING COMPLETE', summarize(), flush=True)


if __name__ == '__main__':
    main()
