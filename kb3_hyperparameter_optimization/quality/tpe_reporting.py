"""Validation reports reuse completed trial scores and trajectories; never retrain."""

import csv
import json
from pathlib import Path

from .detector import atomic_json


def trajectory_diagnostics(record):
    result = record['result']
    path = Path(result['run_dir']) / 'epochs.json'
    history = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    best, final = result['metrics'], result['final_metrics']
    decline = None
    # Describe the observed curve; this alone cannot establish overfitting.
    eligible = [r['metrics'] for r in history if r['metrics']['epoch'] > best['epoch']]
    for start in range(max(0, len(eligible) - 4)):
        window = eligible[start:start + 5]
        if len(window) == 5 and all(r['map50_95'] < best['map50_95'] - .005 for r in window):
            decline = window[0]['epoch']
            break
    return dict(best_epoch=best['epoch'], actual_epochs=result['actual_epochs'],
                final_minus_best_map_points=100 * (final['map50_95'] - best['map50_95']),
                minimum_val_loss_epoch=min(history, key=lambda r: r['metrics']['val_loss'])['metrics']['epoch'] if history else None,
                first_sustained_map_decline_epoch=decline, decline_window=5, decline_threshold_map=.005,
                interpretation='Sustained validation decline is a diagnostic, not proof or an exact onset of overfitting')


def scores(result):
    metric = result['metrics']
    return {**dict(mAP50_95=metric['map50_95'], mAP50=metric.get('map50'), APs=metric.get('ap_small'),
                   precision=metric.get('precision'), AR300=metric.get('recall')),
            **result.get('canonical_best_scores', {})}


def write_evaluation(output, records, reference, settings, *, protocol):
    output = Path(output)
    ranked = sorted(records, key=lambda r: (-r['result']['metrics']['map50_95'], r['number']))
    anchor = records[0]['result']['metrics']['map50_95']
    reference_map = reference['metrics']['mAP50_95'] if reference else None
    rows = []
    for record in records:
        result = record['result']
        row = dict(trial=record['number'], mode=record['proposal']['mode'], seed=result['seed'],
                   parameters=result['hyperparameters'], scores=scores(result),
                   checkpoint=result['best_checkpoint'], checkpoint_sha256=result['checkpoint_sha256'],
                   diagnostics=trajectory_diagnostics(record), early_stopped=result['stopped_early'],
                   elapsed_hours=result['elapsed_seconds'] / 3600,
                   delta_scratch_anchor_map_points=100 * (result['metrics']['map50_95'] - anchor),
                   delta_kb1_pretrained_map_points=100 * (result['metrics']['map50_95'] - reference_map)
                   if reference_map is not None else None)
        rows.append(row)
    document = dict(protocol=protocol, method='A-TPE', metrics_split='validation',
                    requested_trials=len(records), completed_trials=len(records),
                    best_trial=ranked[0]['number'], best_checkpoint=ranked[0]['result']['best_checkpoint'],
                    best_checkpoint_sha256=ranked[0]['result']['checkpoint_sha256'],
                    ranking=[r['number'] for r in ranked], trials=rows, kb1_pretrained_reference=reference,
                    paired_scratch_anchor_trial=0, detector_seed=settings['seed'], independent_detector_seeds=1,
                    extra_training_performed=False, rl_training_runs=0, test_evaluated=False,
                    actual_epochs=sum(r['result']['actual_epochs'] for r in records),
                    detector_hours=sum(r['result']['elapsed_seconds'] for r in records) / 3600,
                    comparison_caveat='Exploratory in-sample validation selection with one detector seed; no proof of general superiority. '
                                      'KB1 pretrained weights have different initialization and historical training/selection budget.')
    report_dir = output / 'reports' / f'trials_{len(records):04d}'
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(report_dir / 'evaluation.json', document)
    atomic_json(output / 'evaluation.json', document)
    columns = ('trial', 'mode', 'seed', 'lr0', 'weight_decay', 'momentum', 'augmentation_strength',
               'mAP50_95', 'mAP50', 'APs', 'precision', 'operating_recall', 'f1', 'AR300',
               'best_epoch', 'actual_epochs', 'early_stopped', 'elapsed_hours',
               'delta_scratch_anchor_map_points', 'delta_kb1_pretrained_map_points', 'checkpoint')
    for target in (report_dir / 'comparison.csv', output / 'comparison.csv'):
        with target.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, row['parameters'].get(key,
                                    row['scores'].get(key, row['diagnostics'].get(key)))) for key in columns})

    def percent(value):
        return '-' if value is None else f'{100 * value:.3f}%'

    lines = [f'# KB3 A–TPE: {len(records)} lượt thử trên validation', '',
             f'Protocol: `{protocol}`. Seed detector chung: {settings["seed"]}.', '',
             'Mỗi lượt train từ YAML kiến trúc, lấy recipe ban đầu từ supervised KB1. '
             'Checkpoint KB1 chỉ được đánh giá, không nạp để train tiếp.', '',
             f'Lượt 0 là mốc scratch đúng tham số KB1; các lượt trước {settings["n_startup_trials"]} '
             f'lấy mẫu khởi tạo; từ lượt {settings["n_startup_trials"]} dùng TPE. '
             'Chỉ tìm lr0/weight_decay, giữ momentum và augmentation theo KB1.', '',
             '| Trial | Cách chọn | mAP50–95 | mAP50 | AP small | Best epoch | Epoch thực | Δ scratch (điểm %) |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["trial"]} | {row["mode"]} | {percent(row["scores"]["mAP50_95"])} | '
                     f'{percent(row["scores"].get("mAP50"))} | {percent(row["scores"].get("APs"))} | '
                     f'{row["diagnostics"]["best_epoch"]} | {row["diagnostics"]["actual_epochs"]} | '
                     f'{row["delta_scratch_anchor_map_points"]:+.3f} |')
    lines += ['', f'Trial được chọn bằng validation: **{document["best_trial"]}**. '
              'Đây là checkpoint tốt nhất trong các lượt đã thử, chưa phải cấu hình tối ưu được chứng minh.', '',
              f'Tổng train: {document["actual_epochs"]} epoch, {document["detector_hours"]:.3f} giờ detector. '
              'Báo cáo không tạo thêm lượt train; chưa đánh giá test.', '']
    if reference:
        lines += [f'Mốc KB1 pretrained: mAP50–95 **{percent(reference_map)}**, '
                  f'mAP50 {percent(reference["metrics"].get("mAP50"))}. '
                  'Cùng evaluator nhưng khác khởi tạo và ngân sách lịch sử; không dùng riêng phép so sánh này '
                  'để quy cải thiện cho TPE.', '']
    lines += ['## Diễn biến train', '',
              'Các mốc dưới đây mô tả đường validation đã quan sát. Mốc mAP giảm liên tiếp không đủ '
              'để khẳng định overfit bắt đầu chính xác tại epoch đó; cần đọc thêm train/val loss và các seed khác.', '']
    for row in rows:
        diagnostic = row['diagnostics']
        lines.append(f'- Trial {row["trial"]}: best epoch {diagnostic["best_epoch"]}, '
                     f'val loss thấp nhất ở {diagnostic["minimum_val_loss_epoch"]}, '
                     f'final − best mAP {diagnostic["final_minus_best_map_points"]:+.3f} điểm %. '
                     f'Mốc giảm mAP ít nhất 0,5 điểm % trong 5 epoch liên tiếp: '
                     f'{diagnostic["first_sustained_map_decline_epoch"]}. Early stop: {row["early_stopped"]}.')
    lines += ['', '## Giới hạn và bước tiếp theo', '',
              f'- {len(records)} lượt gồm một mốc và {len(records)-1} cấu hình mới; '
              f'{sum(r["proposal"]["mode"] == "tpe" for r in records)} proposal được chọn bằng TPE.',
              '- Một seed chung giúp giảm nhiễu khởi tạo, chưa đo được độ ổn định qua nhiều seed.',
              '- Search và chọn best đều dùng validation; điểm thắng có thể lạc quan do chọn nhiều cấu hình/epoch.',
              '- Early stopping có thể bỏ lỡ cấu hình cải thiện muộn; lịch LR vẫn theo horizon tối đa.',
              '- Đây là A–TPE thử nghiệm, chưa có kết quả RL và chưa có so sánh với B.',
              '- Nếu cần mở rộng, dùng --trials với tổng số lượt mong muốn và --resume; giữ cấu hình/source cố định. '
              'Chỉ dùng test sau khi chốt thiết kế và checkpoint.', '']
    plot_path = report_dir / 'trajectories.png'
    if write_plot(plot_path, records):
        lines += ['![Đường train và validation](trajectories.png)', '']
    (report_dir / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'report.md').write_text('\n'.join(lines).replace('(trajectories.png)',
                                    f'(reports/trials_{len(records):04d}/trajectories.png)'), encoding='utf-8')
    print(f'VALIDATION REPORT: {output / "report.md"}; best trial={document["best_trial"]}', flush=True)
    return document


def write_plot(path, records):
    curves = []
    for record in records:
        source = Path(record['result']['run_dir']) / 'epochs.json'
        if source.exists():
            history = json.loads(source.read_text(encoding='utf-8'))
            if history:
                curves.append((record, history))
    if not curves:
        return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for record, history in curves:
        values = [row['metrics'] for row in history]
        x = [row['epoch'] for row in values]
        label = f'Trial {record["number"]}'
        axes[0].plot(x, [100 * row['map50_95'] for row in values], label=label)
        axes[1].plot(x, [row['train_loss'] for row in values], label=label)
        axes[2].plot(x, [row['val_loss'] for row in values], label=label)
    for axis, title in zip(axes, ('Canonical validation mAP50-95 (%)', 'Train loss', 'Validation loss')):
        axis.set_title(title)
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True
