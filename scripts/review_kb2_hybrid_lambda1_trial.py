"""Verify and summarize the one matched lambda1 continuation after completion."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/kb2-matplotlib')
os.environ.setdefault('YOLO_CONFIG_DIR','/tmp')
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.run_kb2_hybrid_lambda1_trial import BASELINE, ROOT, digest, read


METRICS=('mAP50_95','mAP50','mAP75','precision','operating_recall','f1',
         'macro_f1','macro_recall','AR300','APs','APm','APl','tp','fp','fn')


def events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def selector(rows):
    best=rows[0]
    for row in rows[1:]:
        if row['metrics']['mAP50_95']>best['metrics']['mAP50_95']+1e-4:
            best=row
    return best


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def review(root,output,reload_device=None):
    experiment=read(root/'completed.json')
    assert experiment['matched_stream'] and experiment['protected_files_unchanged']
    assert experiment['config_only_lambda_differs']
    protected=experiment['plan']['immutable_files']
    assert all(digest(path)==value for path,value in protected.items())
    output.mkdir(parents=True,exist_ok=False)
    folders={'lambda003':BASELINE/'run/dpo','lambda100':root/'run/dpo'}
    histories={};trains={};completed={}
    for label,folder in folders.items():
        rows=events(folder/'metrics.jsonl')
        histories[label]=[r for r in rows if r['event']=='validation']
        trains[label]=[r for r in rows if r['event']=='train']
        completed[label]=read(folder/'completed.json')
        assert selector(histories[label])['step']==completed[label]['best_step']
        assert completed[label]['steps']==15000
        assert completed[label]['reference_unchanged'] and completed[label]['frozen_state_unchanged']
    assert [r['step'] for r in histories['lambda003']]==[r['step'] for r in histories['lambda100']]
    assert digest(folders['lambda003']/'stream.jsonl')==digest(folders['lambda100']/'stream.jsonl')
    baseline_start=histories['lambda003'][0]['metrics']
    experiment_start=histories['lambda100'][0]['metrics']
    step0_equal=all(baseline_start[k]==experiment_start[k] for k in METRICS)
    assert step0_equal,'Initial measured quality metrics differ; do not claim a matched lambda-only comparison'
    a,b=trains['lambda003'][0],trains['lambda100'][0]
    assert a['native_loss']==b['native_loss'] and a['dpo_loss']==b['dpo_loss']
    gradient_ratio=b['gradient']['weighted_dpo_grad_norm']/a['gradient']['weighted_dpo_grad_norm']
    assert abs(gradient_ratio-(1/.03))<1e-4
    summary=[]
    summary.append({'label':'TPE_step0','step':0,**{k:baseline_start[k] for k in METRICS}})
    selected={}
    for label in folders:
        selected[label]=selector(histories[label])
        for suffix,row in [('best',selected[label]),('last',histories[label][-1])]:
            summary.append({'label':f'{label}_{suffix}','step':row['step'],**{k:row['metrics'][k] for k in METRICS}})
    best_positive=max((r for r in histories['lambda100'] if r['step']>0),key=lambda r:r['metrics']['mAP50_95'])
    summary.append({'label':'lambda100_best_positive_update_by_raw_AP','step':best_positive['step'],
                    **{k:best_positive['metrics'][k] for k in METRICS}})
    write_csv(output/'summary.csv',summary)
    validation=[]
    for label,rows in histories.items():
        validation.extend({'label':label,'step':r['step'],**{k:r['metrics'][k] for k in METRICS}} for r in rows)
    write_csv(output/'validation.csv',validation)
    raw_train=[]
    for label,rows in trains.items():
        weight=.03 if label=='lambda003' else 1.
        raw_train.extend({'label':label,'step':r['step'],'native_loss':r['native_loss'],
                          'dpo_loss':r['dpo_loss'],'weighted_dpo_loss':weight*r['dpo_loss'],
                          'total_loss_reconstructed':r['native_loss']+weight*r['dpo_loss'],
                          'lr':r['lr'],'mean_margin':r['mean_margin']} for r in rows)
    write_csv(output/'training.csv',raw_train)
    deltas={k:selected['lambda100']['metrics'][k]-selected['lambda003']['metrics'][k] for k in METRICS}
    reload_checks={}
    if reload_device is not None:
        import math
        import torch
        from kb2_preference_optimization.conditional.policy import ConditionalAdapter, configure_numerics
        from kb2_preference_optimization.conditional.evaluation import evaluate
        from kb2_preference_optimization.data import detection_loader
        configure_numerics()
        adapter=ConditionalAdapter(experiment['plan']['source']['source_checkpoint'],reload_device)
        loader=detection_loader('valid',640,8)
        source_metrics,source_predictions=evaluate(adapter,loader)
        assert all(math.isclose(source_metrics[k],baseline_start[k],rel_tol=0,abs_tol=1e-7) for k in METRICS)
        trained_manifest=read(folders['lambda100']/'manifest.json')
        for which,logged in [('best',selected['lambda100']['metrics']),('last',histories['lambda100'][-1]['metrics'])]:
            checkpoint_path=folders['lambda100']/f'{which}.pt'
            assert digest(checkpoint_path)==completed['lambda100'][f'{which}_sha256']
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            adapter.model.load_state_dict(checkpoint['state_dict'],strict=True)
            assert adapter.frozen_digest()==trained_manifest['frozen_state_sha256']
            measured,_=evaluate(adapter,loader,source_predictions)
            assert measured['num_images']==479 and measured['num_gt']==1887
            errors={k:abs(measured[k]-logged[k]) for k in METRICS}
            assert all(error<=1e-7 for error in errors.values()),errors
            reload_checks[which]={'matches_logged_metrics':True,'metric_abs_errors':errors,
                                  'step':checkpoint['step'],'checkpoint_sha256':digest(checkpoint_path)}
            (output/f'lambda100_{which}_reload.json').write_text(json.dumps(measured,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps({'event':'checkpoint_reload_verified','which':which,'step':checkpoint['step'],
                              'mAP50_95':measured['mAP50_95']}),flush=True)
        del adapter
        if reload_device.startswith('cuda'):torch.cuda.empty_cache()
    assert all(digest(path)==value for path,value in protected.items())
    report={'experiment_root':str(root),'baseline_root':str(BASELINE),
            'matched_stream':True,'config_only_lambda_differs':True,'step0_metrics_equal':step0_equal,
            'first_batch_native_loss_equal':True,'first_batch_dpo_loss_equal':True,
            'first_batch_weighted_DPO_gradient_ratio':gradient_ratio,
            'first_batch_gradients':{label:trains[label][0]['gradient'] for label in folders},
            'best_steps':{label:row['step'] for label,row in selected.items()},
            'summary':summary,'delta_best_lambda100_minus_lambda003':deltas,
            'delta_units':'fractions for ratio metrics; multiply by100 for percentage points; counts for TP/FP/FN',
            'checkpoint_reloads':reload_checks,
            'protected_files_unchanged':True,'test_evaluated':False,'selected_model_promotion':False,
            'scope':'One matched continuation seed42, development validation; recipe starts from TPE trial2.'}
    (output/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    colors={'lambda003':'#237A57','lambda100':'#B76532'}
    labels={'lambda003':'λ=0,03','lambda100':'λ=1'}
    for label,rows in histories.items():
        xx=[r['step'] for r in rows]
        yy=[r['metrics']['mAP50_95']*100 for r in rows]
        axes[0,0].plot(xx,yy,'o-',ms=3,lw=1.3,color=colors[label],label=labels[label])
        early=[r for r in rows if r['step']<=2000]
        axes[0,1].plot([r['step'] for r in early],[r['metrics']['mAP50_95']*100 for r in early],
                       'o-',ms=4,color=colors[label],label=labels[label])
        axes[1,0].plot(xx,[r['metrics']['precision']*100 for r in rows],color=colors[label],label=labels[label]+' precision')
        axes[1,0].plot(xx,[r['metrics']['operating_recall']*100 for r in rows],ls='--',color=colors[label],label=labels[label]+' recall')
        loss=trains[label]
        axes[1,1].plot([r['step'] for r in loss],[r['dpo_loss'] for r in loss],alpha=.25,color=colors[label])
        import pandas as pd
        axes[1,1].plot([r['step'] for r in loss],pd.Series([r['dpo_loss'] for r in loss]).rolling(15,min_periods=1).mean(),
                       color=colors[label],label=labels[label]+' DPO, rolling15 records')
    for ax in axes[0]:
        ax.axhline(100*baseline_start['mAP50_95'],ls=':',color='#7B8490',label='TPE step0')
        ax.set_ylabel('Canonical mAP50–95 (%)')
    axes[0,0].set_title('Toàn run: cùng source/bank/stream/schedule')
    axes[0,1].set_title('Giai đoạn đầu:0–2000 update')
    axes[1,0].set_title('Precision / recall tại conf0,25')
    axes[1,0].set_ylabel('Metric (%)')
    axes[1,1].set_title('DPO loss chưa nhân λ')
    axes[1,1].set_ylabel('DPO loss')
    for ax in axes.flat:
        ax.set_xlabel('Optimizer update');ax.grid(alpha=.18);ax.legend(fontsize=7)
    fig.suptitle('Hybrid KB2: λ=1 so với λ=0,03 — seed42',fontweight='bold')
    fig.tight_layout(rect=(0,0,1,.96))
    for extension in ('png','pdf','svg'):fig.savefig(output/f'lambda_comparison.{extension}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    best_old=selected['lambda003'];best_new=selected['lambda100']
    lines=[
        '# Hybrid KB2 — thử λ=1 so với λ=0,03','',
        'Giữ TPE trial2, bank773 cặp, seed42, background0,5, β0,1, AdamW LR1e−6; '
        'LR giảm sau1000, eval mỗi250 tới2000 rồi mỗi1000, budget15000 update.',
        'Stream ảnh/pair được xác minh giống hệt; chỉ dpo_weight thay đổi. Không đánh giá test hoặc thay model chốt.','',
        '| Checkpoint | Step | mAP50–95 (%) | AP50 (%) | AP75 (%) | Precision (%) | Recall (%) | F1 (%) | TP | FP | FN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary:
        values=[row['label'],str(row['step'])]+[f"{100*row[k]:.4f}" for k in ('mAP50_95','mAP50','mAP75','precision','operating_recall','f1')]+[str(row[k]) for k in ('tp','fp','fn')]
        lines.append('| '+' | '.join(values)+' |')
    lines += ['',f"Best λ1 ở step{best_new['step']}; so best λ0,03 ở step{best_old['step']}: "
              f"ΔmAP={100*deltas['mAP50_95']:+.4f} điểm phần trăm, "
              f"ΔP={100*deltas['precision']:+.4f}, ΔR={100*deltas['operating_recall']:+.4f}, "
              f"ΔF1={100*deltas['f1']:+.4f}; FP{deltas['fp']:+d}, FN{deltas['fn']:+d}.",
              '', f"Ở cùng update15000, ΔmAP λ1−λ0,03 = "
              f"{100*(histories['lambda100'][-1]['metrics']['mAP50_95']-histories['lambda003'][-1]['metrics']['mAP50_95']):+.4f} điểm phần trăm. "
              'Do đó λ0,03 nhỉnh hơn theo selected-best, trong khi λ1 cao hơn về mAP khi chỉ so last15k.',
              '', ('Đã nạp lại best/last λ1 và chấm đủ479 ảnh/1887GT; các metric khớp log validation, '
                   'đồng thời xác nhận frozen-state và checkpoint SHA256.' if reload_checks else
                   'Bảng dùng validation log của checkpoint đã lưu; chưa yêu cầu reload trong lệnh review này.'),
              '', '| Batch đầu, cùng model/ảnh/pair | λ=0,03 | λ=1 |',
              '|---|---:|---:|',
              f"| Native loss | {a['native_loss']:.6f} | {b['native_loss']:.6f} |",
              f"| λ × DPO loss | {.03*a['dpo_loss']:.6f} | {b['dpo_loss']:.6f} |",
              f"| Native gradient norm | {a['gradient']['native_grad_norm']:.6f} | {b['gradient']['native_grad_norm']:.6f} |",
              f"| Weighted-DPO gradient norm | {a['gradient']['weighted_dpo_grad_norm']:.6f} | {b['gradient']['weighted_dpo_grad_norm']:.6f} |",
              f"| Weighted-DPO/native norm ratio | {100*a['gradient']['weighted_dpo_native_norm_ratio']:.4f}% | {100*b['gradient']['weighted_dpo_native_norm_ratio']:.4f}% |",
              '',f'Batch đầu: weighted-DPO norm λ1/λ0,03={gradient_ratio:.6f}, '
              'khớp tỷ lệ1/0,03 trước khi cộng native gradient/clipping/AdamW. Không coi đó là tỷ lệ update weights.',
              '', 'Selector cho phép TPE step0 thắng nếu continuation không cải thiện đủ min_delta1e−4. '
              'Dòng best_positive_update chỉ là AP cao nhất trong các checkpoint đã train, không thay selector.',
              '', 'Một seed continuation và validation phát triển; kết quả chưa xác lập λ tối ưu phổ quát.',
              '', '![So sánh theo update](assets/kb2_hybrid_lambda1_20261113/lambda_comparison.png)',
              '', 'Artifact: [review.json](assets/kb2_hybrid_lambda1_20261113/review.json), '
              '[summary.csv](assets/kb2_hybrid_lambda1_20261113/summary.csv).']
    report_path=ROOT/'docs/report_final/KB2_HYBRID_LAMBDA1_REVIEW_20261113.md'
    report_path.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'review':str(report_path),'best_steps':report['best_steps'],
                      'delta_best_mAP_pp':100*deltas['mAP50_95'],'matched_stream':True},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/report_final/assets/kb2_hybrid_lambda1_20261113')
    parser.add_argument('--reload-device',default=None,help='Optional fresh best/last validation, e.g. cuda or cpu')
    args=parser.parse_args()
    review(args.root.resolve(),args.output.resolve(),args.reload_device)
