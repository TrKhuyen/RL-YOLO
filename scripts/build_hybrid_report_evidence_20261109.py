"""Read-only audit/tables/figures for the complete KB3-TPE -> KB2-DPO report."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a fresh evidence directory')
    tpe = REPO/'kb3_hyperparameter_optimization/checkpoint_hyperparameter_optimization/quality/tpe_kb1_pretrained_v2/yolov8n'
    eval_ = read(tpe/'evaluation.json')
    manifest = read(tpe/'manifest.json')
    assert eval_['completed_trials'] == eval_['requested_trials'] == 8
    assert eval_['best_trial'] == 2 and not eval_['test_evaluated'] and eval_['rl_training_runs'] == 0
    protected = {str(tpe/'evaluation.json'): digest(tpe/'evaluation.json'),
                 str(tpe/'manifest.json'): digest(tpe/'manifest.json')}
    trials, initial_hashes = [], set()
    for i in range(8):
        folder = tpe/'tpe'/f'trial_{i:04d}'
        r, spec = read(folder/'result.json'), read(folder/'trial_spec.json')
        path = folder/'selected_best.pt'
        assert digest(path) == r['checkpoint_sha256']
        assert r['initialization'] == 'pretrained' and r['pretrained_backbone_verified']
        assert r['seed'] == 0 and r['hyperparameters'] == spec['parameters']
        assert r['pretrained_source'] == manifest['settings']['pretrained_source']
        protected[str(path)] = digest(path)
        protected[str(folder/'result.json')] = digest(folder/'result.json')
        initial_hashes.add(r['initial_weights_sha256'])
        score = r['canonical_best_scores']
        trials.append({'trial': i, 'mode': spec['mode'], 'sampler_seed': spec['sampler_seed'],
                       **r['hyperparameters'], 'native_selected_epoch': r['metrics']['epoch'],
                       'actual_epochs': r['actual_epochs'], 'detector_hours': r['elapsed_seconds']/3600,
                       **{key: score[key] for key in ('mAP50_95', 'mAP50', 'precision', 'operating_recall', 'f1')},
                       'last_mAP50_95': r['canonical_final_scores']['mAP50_95'],
                       'checkpoint_sha256': r['checkpoint_sha256']})
    assert len(initial_hashes) == 1 and sum(r['actual_epochs'] for r in trials) == 1387
    assert math.isclose(sum(r['detector_hours'] for r in trials), eval_['detector_hours'])
    assert max(trials, key=lambda r:r['mAP50_95'])['trial'] == 2
    payload = torch.load(tpe/'tpe/trial_0002/selected_best.pt', weights_only=False, map_location='cpu')
    model = payload.get('ema') or payload['model']
    architecture = {'parameters': sum(p.numel() for p in model.parameters()),
                    'DPO_trainable_parameters': sum(p.numel() for p in model.model[-1].cv3.parameters()),
                    'DPO_trainable_tensors': len(list(model.model[-1].cv3.parameters())),
                    'nc': model.model[-1].nc, 'reg_max': model.model[-1].reg_max,
                    'strides': model.model[-1].stride.tolist(), 'yaml': model.yaml}
    selection_dir = REPO/'docs/report_final/assets/kb2_hybrid_selected_20261108'
    selection = read(selection_dir/'selection.json')
    assert selection['selected_update'] == 500 and selection['lambda'] == .03
    assert digest(selection['checkpoint']) == selection['checkpoint_sha256']
    protected[selection['checkpoint']] = selection['checkpoint_sha256']
    assert selection['initialization_and_reference']['source_sha256'] == trials[2]['checkpoint_sha256']
    schedule = REPO/'docs/report_final/assets/kb2_hybrid_schedule_20261106'
    metrics = {label:read(schedule/f'{name}.json') for label,name in
               [('S0','S0'),('TPE','TPE'),('hybrid_best','scheduled_lambda003_best'),
                ('hybrid_last','scheduled_lambda003_last'),('lambda010_best','constant_lambda010_best'),
                ('constant_lambda003_best','constant_lambda003_best')]}
    fields = ('mAP50_95','mAP50','mAP75','precision','operating_recall','f1','macro_f1',
              'macro_recall','AR300','APs','APm','APl','tp','fp','fn')
    metric_rows = [{'model':label, **{k:m[k] for k in fields}} for label,m in metrics.items()]
    for label,m in metrics.items():
        assert m['num_images'] == 479 and m['num_gt'] == 1887
        assert math.isclose(m['precision'],m['tp']/(m['tp']+m['fp']),abs_tol=1e-12)
        assert m['tp']+m['fn']==1887
        protected[str(schedule/f"{dict(S0='S0',TPE='TPE',hybrid_best='scheduled_lambda003_best',hybrid_last='scheduled_lambda003_last',lambda010_best='constant_lambda010_best',constant_lambda003_best='constant_lambda003_best')[label]}.json")] = digest(schedule/f"{dict(S0='S0',TPE='TPE',hybrid_best='scheduled_lambda003_best',hybrid_last='scheduled_lambda003_last',lambda010_best='constant_lambda010_best',constant_lambda003_best='constant_lambda003_best')[label]}.json")
    bank = Path(selection['config'].get('bank_path', REPO/'kb2_preference_optimization/outputs/yolov8n_hybrid_tpe2_s42_20261105_180813_658387/preferences'))
    bank_manifest = read(bank/'manifest.json')
    assert bank_manifest['reference_state_sha256']
    assert bank_manifest['audit']['totals']['pairs'] == 773
    for name in ('manifest.json','pairs.jsonl','contexts.jsonl'):
        protected[str(bank/name)] = digest(bank/name)
    run = Path(selection['checkpoint']).parent
    rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    vals = [r for r in rows if r['event']=='validation']
    assert len(vals)==22 and max(vals,key=lambda r:r['metrics']['mAP50_95'])['step']==500
    old = REPO/'docs/report_final/assets/kb2_strength_multiseed_20261103'
    historical = list(csv.DictReader((old/'summary.csv').open(encoding='utf-8-sig')))
    paired = list(csv.DictReader((old/'paired_deltas.csv').open(encoding='utf-8-sig')))
    out.mkdir(parents=True,exist_ok=False)
    csv_write(out/'tpe_trials.csv',trials)
    csv_write(out/'current_metrics.csv',metric_rows)
    csv_write(out/'dpo_validation.csv',[{'step':r['step'],**{k:r['metrics'][k] for k in fields}} for r in vals])
    csv_write(out/'historical_S0_controls.csv',historical)
    csv_write(out/'historical_paired_deltas.csv',paired)
    fig,ax=plt.subplots(1,2,figsize=(12,4.5))
    scores=[100*r['mAP50_95'] for r in trials]
    ax[0].plot(range(8),scores,'o-',label='Canonical score of native-selected best')
    ax[0].plot(range(8),[max(scores[:i+1]) for i in range(8)],'s--',label='Running best')
    ax[0].set(xlabel='Logical trial',ylabel='Historical canonical mAP50-95 (%)',title='TPE search: 8 detector trials, seed0')
    for i,v in enumerate(scores):ax[0].annotate(f'{v:.3f}',(i,v),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8)
    ax[0].legend(fontsize=8)
    scatter=ax[1].scatter([r['lr0'] for r in trials],[r['weight_decay'] for r in trials],c=scores,cmap='viridis',s=80)
    for r in trials:ax[1].annotate(str(r['trial']),(r['lr0'],r['weight_decay']),xytext=(5,5),textcoords='offset points')
    ax[1].set(xscale='log',yscale='log',xlabel='lr0',ylabel='Input weight_decay',title='Tested local log-space configurations')
    fig.colorbar(scatter,ax=ax[1],label='Historical canonical mAP (%)')
    for a in ax:a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'tpe_search.png',dpi=160);fig.savefig(out/'tpe_search.svg');plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4.3))
    for a,subset,title in [(ax[0],[r for r in vals if r['step']<=2000],'Early 0-2k, dense evaluation'),(ax[1],vals,'Complete 15k budget')]:
        a.plot([r['step'] for r in subset],[100*r['metrics']['mAP50_95'] for r in subset],'o-',label='KB2 pipeline, lambda=.03')
        a.axhline(100*metrics['TPE']['mAP50_95'],ls=':',color='black',label='TPE initialization')
        a.scatter([500],[100*metrics['hybrid_best']['mAP50_95']],s=110,marker='*',color='red',label='Selected best500',zorder=5)
        a.axvline(1000,ls='--',color='gray',label='LR drops after1000')
        a.set(xlabel='Optimizer updates',ylabel='Current validation mAP50-95 (%)',title=title);a.legend(fontsize=8);a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'dpo_selection.png',dpi=160);fig.savefig(out/'dpo_selection.svg');plt.close(fig)
    evidence={'date_local':'2026-11-09','timezone':'Asia/Bangkok','no_new_training_or_model_evaluation':True,
              'TPE_evaluation':{k:v for k,v in eval_.items() if k not in ('trials','kb1_pretrained_reference')},
              'TPE_settings':manifest['settings'],'TPE_dependencies':manifest['dependencies'],
              'TPE_trial_rows':trials,'same_initial_hash_all_8_trials':next(iter(initial_hashes)),
              'architecture':architecture,'selection':selection,'bank_audit':bank_manifest['audit'],
              'current_metrics':metric_rows,'protected_files':protected,
              'historical_control_scope':'S0 initialization, lambda .1, seeds42/43/44; not hybrid confirmation'}
    for filename in ('selection.json','mechanism_evidence.json','clarification_examples_20261109.json'):
        protected[str(selection_dir/filename)]=digest(selection_dir/filename)
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    assert all(digest(path)==sha for path,sha in protected.items())
    print(json.dumps({'output':str(out),'trials_verified':8,'native_epoch_winner':139,
                      'total_epochs':1387,'hours':eval_['detector_hours'],'architecture':{k:v for k,v in architecture.items() if k!='yaml'},
                      'protected_files':len(protected),'current_mAP':metrics['hybrid_best']['mAP50_95']}))


if __name__=='__main__':
    main()
