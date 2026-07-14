#!/usr/bin/env python3
import argparse, json, os
p=argparse.ArgumentParser(); p.add_argument('--run-dir', required=True); a=p.parse_args()
def load(path):
    with open(path, encoding='utf-8') as f: return json.load(f)
items=[('Stage 0 sensitivity','stage0/summaries/stage0_gate.json'),('ACC effectiveness','stage1/summaries/effectiveness.json'),('Forgetting and SOR','stage2/summaries/forgetting.json')]
lines=['# CoPTER Experiment Report','', 'Single-seed experiment. Results are reproducible evidence, not a statistical-significance claim.','']
all_pass=True
for title, rel in items:
    path=os.path.join(a.run_dir,rel); data=load(path); passed=bool(data.get('pass', data.get('passed', data.get('gate_passed', False)))); all_pass &= passed
    lines += ['## '+title, '', 'Gate: **%s**' % ('PASS' if passed else 'FAIL'), '', '```json', json.dumps(data,ensure_ascii=False,indent=2), '```', '']
lines.insert(2, 'Overall: **%s**' % ('PASS' if all_pass else 'FAIL'))
out=os.path.join(a.run_dir,'experiment_report.md')
with open(out+'.tmp','w',encoding='utf-8') as f: f.write('\n'.join(lines))
os.replace(out+'.tmp',out)
raise SystemExit(0 if all_pass else 2)
