"""Reload paired checkpoints, verify recorded metrics, and report all locked contrasts."""
import argparse
import csv
import json
import math
from pathlib import Path

from scripts.run_kb2_paired_control import configure, read, verify_pair, FIELDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); configure()
    import torch
    from kb2_preference_optimization.artifacts import atomic_json, sha256
    from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
    from kb2_preference_optimization.conditional.evaluation import evaluate
    from kb2_preference_optimization.data import detection_loader
    root=args.campaign.resolve();out=args.output.resolve()
    done=read(root/'completed.json');plan=done['plan']
    assert all(sha256(Path(p))==h for p,h in plan['protected'].items())
    records=[]
    for seed in plan['seeds']:records.extend(verify_pair(root,seed))
    assert records==done['paired_results']['records']
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'plan.json',plan)
    atomic_json(out/'paired_results.json',done['paired_results'])
    (out/'paired_results.csv').write_bytes((root/'paired_results.csv').read_bytes())
    configure_numerics();loader=detection_loader('valid',640,8)
    adapter=ConditionalAdapter(plan['source']['source_checkpoint'],'cuda')
    reloaded=[];trajectory=[];manifests={}
    for seed in plan['seeds']:
        for arm in ('native','dpo'):
            folder=root/f'seed{seed}'/arm
            manifests[f'{seed}_{arm}']=read(folder/'manifest.json')
            for line in (folder/'metrics.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row['event']=='validation':
                    trajectory.append({'seed':seed,'arm':arm,'step':row['step'],**{k:row['metrics'][k] for k in FIELDS}})
        for row in [r for r in records if r['seed']==seed]:
            predictions=None;pair={}
            for arm in ('native','dpo'):
                step=row[f'{arm}_step'];folder=root/f'seed{seed}'/arm
                checkpoint=folder/f'step_{step:06d}.pt'
                state=torch.load(checkpoint,weights_only=False,map_location='cpu')
                adapter.model.load_state_dict(state['state_dict'],strict=True)
                assert adapter.frozen_digest()==state['manifest']['frozen_state_sha256']
                metrics,cached=evaluate(adapter,loader,predictions)
                if arm=='native':predictions=cached
                assert metrics['num_images']==479 and metrics['num_gt']==1887
                for key in FIELDS:
                    assert math.isclose(metrics[key],state['metrics'][key],rel_tol=0,abs_tol=1e-7),(seed,arm,step,key)
                assert metrics['per_class']==state['metrics']['per_class']
                atomic_json(out/f"seed{seed}_{row['contrast']}_{arm}.json",metrics)
                pair[arm]=metrics
                reloaded.append({'seed':seed,'contrast':row['contrast'],'arm':arm,'step':step,'sha256':sha256(checkpoint),'metrics_match_log':True})
            print(json.dumps({'event':'reloaded_pair','seed':seed,'contrast':row['contrast'],
                              'delta_mAP_pp':100*(pair['dpo']['mAP50_95']-pair['native']['mAP50_95']),
                              'GT_transitions_DPO_vs_native':pair['dpo']['source_gt_transitions']}),flush=True)
    atomic_json(out/'verification.json',{'reloaded_checkpoints':reloaded,'paired_manifests':manifests,
                                       'all_streams_matched_within_seed':True,'all_protected_hashes_unchanged':all(sha256(Path(p))==h for p,h in plan['protected'].items())})
    with (out/'validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(trajectory[0]),lineterminator='\n');w.writeheader();w.writerows(trajectory)
    labels={'step500':'Cùng step500 — tiêu chí chính','selected_best':'Best theo cùng selector/grid','step15000':'Cùng step15000'}
    lines=['# Đối chứng ghép cặp native/DPO từ TPE — 16/11/2026','',
           'Sáu lượt chạy mới: seed42/43/44 × λ0/λ0,03. Cùng TPE trial2, bank, stream trong mỗi seed, mask cv3, native background0,5, AdamW LR1e−6; LR giảm10 lần từ update1001; eval mỗi250 tới2000 rồi mỗi1000;15.000 update mỗi lượt.',
           '', 'Tiêu chí chính khóa trước: Δ mAP tại step500. Best và last15000 là tiêu chí bổ sung. Best dùng min_delta1e−4 fraction và cho phép step0; không thay tiêu chí sau khi xem số liệu.',
           '', '| Phép so | Seed | Step native / DPO | Native mAP (%) | DPO mAP (%) | Δ DPO−native, điểm % | Δ FP | Δ FN |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {labels[r['contrast']]} | {r['seed']} | {r['native_step']} / {r['dpo_step']} | {100*r['native_mAP50_95']:.6f} | {100*r['dpo_mAP50_95']:.6f} | {r['delta_mAP50_95']:+.6f} | {r['delta_fp']:+d} | {r['delta_fn']:+d} |")
    lines+=['','| Phép so | Δ mAP trung bình ± SD mẫu, điểm % | Seed DPO cao hơn |','|---|---:|---:|']
    for contrast,s in done['paired_results']['summary'].items():
        lines.append(f"| {labels[contrast]} | {s['mean_delta_mAP_pp']:+.6f} ± {s['sample_sd_delta_mAP_pp']:.6f} | {s['positive_seeds']}/3 |")
    s=done['paired_results']['summary']['step500']
    lines+=['','## Diễn giải', '',
        f"Tại step500, DPO cao hơn native ở {s['positive_seeds']}/3 seed, thấp hơn ở {s['negative_seeds']}/3 seed, Δ trung bình {s['mean_delta_mAP_pp']:+.6f} điểm phần trăm.",
        'Đây là đối chứng can thiệp trọng số DPO trong recipe cố định; chênh lệch ghép cặp kiểm tác dụng thêm DPO trên validation này. Không quy toàn bộ gain hybrid−TPE cho DPO: native continuation là đối chứng cần so.',
        'Nếu dấu thay đổi giữa seed hoặc mức tăng rất nhỏ, không khẳng định lợi ích DPO ổn định. Nếu DPO thắng cả ba seed, bằng chứng chỉ áp dụng cho ba continuation seed từ một TPE và validation đã dùng phát triển; chưa chứng minh tổng quát hóa độc lập hoặc ưu thế mọi metric.',
        'Seed42 và step500 từng được dùng phát triển recipe; khóa phép so trước campaign mới không biến validation này thành holdout mới. SD là độ lệch chuẩn mẫu qua ba chênh lệch ghép cặp, không phải khoảng tin cậy hoặc kiểm định ý nghĩa thống kê.',
        '', '## Kiểm chứng', '',
        'Đã kiểm lại config/stream/hash, selector và lịch LR của sáu run. Gradient auxiliary của native bằng0 ở các mốc diagnostic. Đã nạp lại18 checkpoint theo ba phép so × ba seed × hai arm và đánh giá đủ479 ảnh/1887GT; metric chính và per-class khớp log. Frozen-state giữ nguyên. File DPO reloaded có source_gt_transitions so trực tiếp với native tại cùng phép so.',
        '', '[Kết quả máy đọc được](paired_results.json), [CSV](paired_results.csv), [xác minh](verification.json), [quỹ đạo](validation.csv), [kế hoạch khóa trước](plan.json).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'event':'review_completed','report':str(out/'REPORT.md')}),flush=True)


if __name__=='__main__':main()
