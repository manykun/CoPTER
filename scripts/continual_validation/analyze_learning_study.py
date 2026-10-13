#!/usr/bin/env python3
"""Descriptive acquisition, objective alignment and retention, never a forced gate."""
import argparse
import csv
import sys
from pathlib import Path
from analyze_acc_attribution import analyze, export, read
from rescore_rewards import score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'copter'))
from structures import acc_action_from_indices


def expected_nodes(p, scope):
    a, b = p['manifest']['task_a'], p['manifest']['task_b']
    nodes = [('reference_mid', a), ('reference_mid', b)]
    for profile in p['profiles']:
        nodes += [(f'{profile}_a_{n}', a) for n in [0]+p['a_points']]
        nodes += [(f'{profile}_a_{p["a_points"][-1]}', b)]
        if scope == 'all':
            for arm in ['aa','ab','return']:
                points = p['return_points'] if arm == 'return' else p['b_points']
                for n in points:
                    nodes += [(f'{profile}_{arm}_{n}', a)]
                    if arm != 'aa':
                        nodes += [(f'{profile}_{arm}_{n}', b)]
    return nodes


def validate_matrix(root, scope):
    p = read(root/'protocol.json')
    for label, task in expected_nodes(p, scope):
        directory = root/'eval'/label/task/'repeat_1'
        for name in ['metrics.json','complete.json']:
            if not (directory/name).is_file():
                raise ValueError(f'Incomplete {scope}: {directory/name}')
    return p


def relative(before, after):
    return (after-before)/abs(before) if before is not None and after is not None and abs(before)>1e-12 else None


def recovery(a, b, returned, lower_better=False):
    loss = b-a if lower_better else a-b
    if loss <= 1e-12:
        return None  # no initial degradation: no meaningful recovery fraction
    return (b-returned if lower_better else returned-b)/loss


def tvd(left, right):
    nl, nr = sum(left.values()), sum(right.values())
    if not nl or not nr:
        return None
    return .5*sum(abs(left.get(k,0)/nl-right.get(k,0)/nr) for k in set(left)|set(right))


def changes(before, after):
    return dict(reward_change=relative(before['reward'], after['reward']),
                p95_change=relative(before['p95_fct_us'],after['p95_fct_us']),
                completion_change_pp=100*(after['completion_ratio']-before['completion_ratio']))


def fmt(x, percent=False):
    return 'n/a' if x is None else (f'{100*x:+.2f}%' if percent else f'{x:.6g}')


def analyze_study(root, scope='all'):
    root = Path(root)
    p = validate_matrix(root, scope)
    out = root/('acquisition_report' if scope == 'acquire' else 'study_report')
    raw = out/'attribution_report'
    analyze(root, raw, p['ports'])
    data = read(raw/'analysis.json')['network']
    indexed = {(r['label'],r['task']):r for r in data}
    a,b = p['manifest']['task_a'],p['manifest']['task_b']
    comparisons, actions, dual = [], [], []
    with (raw/'reward_components.csv').open() as f:
        for r in csv.DictReader(f):
            profile = 'weighted' if r['label'].startswith('weighted_') else 'tail_safe'
            dual.append(dict(r, **score(r,[.5,.3,.2],profile)))
    dual_network={(r['label'],r['task']):r for r in dual if r['port']=='network'}
    def compare(profile, kind, task, before_label, after_label):
        before,after = indexed[before_label,task],indexed[after_label,task]
        row = dict(profile=profile, comparison=kind, task=task, before=before_label,
                   after=after_label, **changes(before,after))
        # The fixed reference was recorded once under tail_safe. Compare it
        # with weighted-trained policies using its SAME-trajectory weighted
        # score, never subtract raw native scores across formulas.
        key='weighted_score' if profile=='weighted' else 'tail_safe_score'
        row['reward_change']=relative(dual_network[before_label,task][key],dual_network[after_label,task][key])
        for key in ['weighted_score','tail_safe_score']:
            db,da=dual_network[before_label,task][key],dual_network[after_label,task][key]
            row[key+'_change']=relative(db,da)
            row[key+'_up_with_p95_worse'] = (da>db and after['p95_fct_us']>before['p95_fct_us']) if db is not None and da is not None else None
        comparisons.append(row)
        l = read(root/'eval'/before_label/task/'repeat_1/metrics.json').get('watch_ports_metrics',{})
        r = read(root/'eval'/after_label/task/'repeat_1/metrics.json').get('watch_ports_metrics',{})
        for port in p['ports']:
            left,right = l.get(str(port),{}),r.get(str(port),{})
            lc,rc = left.get('actions_all',{}),right.get('actions_all',{})
            item = dict(profile=profile,comparison=kind,task=task,port=port,
                        before=before_label,after=after_label,tvd=tvd(lc,rc),
                        before_samples=sum(lc.values()),after_samples=sum(rc.values()))
            for side, counts in [('before',lc),('after',rc)]:
                dominant = max(counts,key=counts.get) if sum(counts.values()) else None
                item[side+'_dominant'] = dominant
                item[side+'_fraction'] = counts[dominant]/sum(counts.values()) if dominant else None
                if dominant and dominant != 'uncontrolled':
                    param = acc_action_from_indices(dominant.split(','),p['manifest']['action_space'])
                    item[side+'_kmin_norm'] = param.k_min_norm
                    item[side+'_kmax_norm'] = param.k_max_norm
                    item[side+'_pmax'] = param.p_max
            actions.append(item)
        return row
    for profile in p['profiles']:
        a0,ae = f'{profile}_a_0',f'{profile}_a_{p["a_points"][-1]}'
        previous = a0
        for n in p['a_points']:
            label=f'{profile}_a_{n}'
            compare(profile,'A_interval',a,previous,label)
            previous=label
        compare(profile,'A_acquisition',a,a0,ae)
        compare(profile,'A_vs_fixed_reference',a,'reference_mid',ae)
        if scope == 'all':
            previous = ae
            for n in p['b_points']:
                ab,aa=f'{profile}_ab_{n}',f'{profile}_aa_{n}'
                compare(profile,'B_interval',b,previous,ab)
                compare(profile,'B_acquisition',b,ae,ab)
                compare(profile,'forgetting_on_A',a,ae,ab)
                compare(profile,'AB_vs_matched_AA',a,aa,ab)
                previous=ab
            be=f'{profile}_ab_{p["b_points"][-1]}'
            compare(profile,'B_vs_fixed_reference',b,'reference_mid',be)
            for n in p['return_points']:
                label=f'{profile}_return_{n}'
                row=compare(profile,'return_A_vs_after_B',a,be,label)
                for key,lower in [('reward',False),('p95_fct_us',True)]:
                    row[key+'_recovery']=recovery(indexed[ae,a][key],indexed[be,a][key],indexed[label,a][key],lower)
                compare(profile,'return_A_residual_vs_after_A',a,ae,label)
                compare(profile,'return_A_cost_on_B',b,be,label)
    export(out/'comparisons.csv',comparisons)
    export(out/'action_changes.csv',actions)
    export(out/'dual_scores.csv',dual)
    lines=['# ACC learning / reward / retention study','',f'Scope: {scope}; seed: {p["manifest"]["seed"]}.',
           'Same-task common completed-flow intersection across all currently available checkpoints; completion includes all offered flows.',
           'Acquisition and final reports may have different intersections: compare rows within the same report.',
           'No PASS/FAIL and no convergence claim from loss alone. Each interval is a finite-budget observation, not proof of asymptotic convergence.',
           'Native reward differences across profiles are not comparable. dual_scores.csv scores the same observations under both formulas.',
           'Fixed reference = predeclared action 4,3, not an optimized policy or a paper SECN baseline.','',
           '| Profile | Comparison | Task | Before → after | Reward change | p95 change | Completion change pp |',
           '|---|---|---|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f'| {r["profile"]} | {r["comparison"]} | {r["task"]} | {r["before"]} → {r["after"]} | '
                     f'{fmt(r["reward_change"],True)} | {fmt(r["p95_change"],True)} | {fmt(r["completion_change_pp"])} |')
    lines += ['', '## How to interpret',
        '- Acquisition: require physical improvement versus initialization and examine the fixed reference, completion, and adjacent checkpoints together.',
        '- Small changes over the last interval suggest a plateau at the measured resolution, not guaranteed convergence; poor plateau performance is not successful learning.',
        '- Reward direction bias: inspect within-task score changes against FCT and completion, including fixed-population port all means. ECN increases alone are not a failure.',
        '- Forgetting: compare frozen A after B with after A, and AB with matched AA; compare action distributions on A with A, not A with B.',
        '- action_changes.csv gives total variation distance (0=same distribution, 1=disjoint), dominant normalized parameters and sample counts. State visitation still differs, so this is not a fixed-state neural-policy test.',
        '- Return-A: recovery is reported only when the metric originally deteriorated. Remaining deficit after the finite return budget is not proof of irreversible forgetting.',
        '- Full local replay is retained in AA, AB and return-A; global replay is off. This tests the algorithm including its normal replay retention.',
        '- Loss and batch reward are training signals; frozen reward is separate. Missing PFC in attribution_report/physical_ports.csv stays n/a.',
        '- A single seed is mechanism evidence, not a significance/generalization claim. No unfavorable checkpoint is removed.']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        lines += ['', 'Install matplotlib to render curves; all numeric CSVs are available.']
    else:
        fig,axes=plt.subplots(3,2,figsize=(13,11))
        for profile in p['profiles']:
            for col,task in enumerate([a,b]):
                segments=[('a',[0]+p['a_points'])] if task==a else [('a',[p['a_points'][-1]])]
                if scope=='all':
                    segments += [('ab',p['b_points']),('return',p['return_points'])]
                labels=[f'{profile}_{arm}_{n}' for arm,ns in segments for n in ns]
                for ax,key in [(axes[0,col],'weighted_score'),(axes[1,col],'tail_safe_score'),(axes[2,col],'p95_fct_us')]:
                    values=[indexed[l,task][key] if key=='p95_fct_us' else dual_network[l,task][key] for l in labels]
                    ax.plot(range(len(labels)),values,'o-',label=profile+' trained')
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels([l[len(profile)+1:] for l in labels],rotation=45)
                    ax.set(title=task,ylabel='Frozen '+key)
                    ax.legend(); ax.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(out/'frozen_curves.png',dpi=150); plt.close(fig)
        lines += ['', 'Frozen curves: frozen_curves.png. Training loss/batch/online curves: attribution_report/training_pPORT.png.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(out/'REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--scope',choices=['acquire','all'],default='all')
    args=parser.parse_args()
    analyze_study(args.run_dir,args.scope)
