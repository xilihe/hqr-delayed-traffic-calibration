"""Render revision tables directly from verified result files, retaining source hashes."""
from pathlib import Path
import csv,hashlib,json,statistics
R=Path(__file__).resolve().parents[3];V=R/'revisions/major-v1';D=V/'results';M=V/'manuscript';T=M/'tables';T.mkdir(exist_ok=True)
source_map=[]
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
def pct(v):return f'{float(v)*100:.2f}'
def esc(s):return str(s).replace('_',r'\_').replace('&',r'\&').replace('%',r'\%')
def save(name,text,inputs):
 
 for a,b in {'use32': 'use 32', 'paired28': 'paired 28', 'day95': 'day 95', 'All53': 'All 53', 'leaves19': 'leaves 19', 'partition20261122': 'partition 20261122', 'all7/28': 'all 7/28', 'under7/28': 'under 7/28', 'Complete17': 'Complete 17', 'All30': 'All 30'}.items():text=text.replace(a,b)
 p=T/(name+'.tex');p.write_text(text);source_map.append(dict(output=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),inputs=[dict(path=str(x.relative_to(R)),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in inputs]))
def main_table(name,caption,label,headers,rows,inputs):
 text=r'\begin{table}[tbp]'+'\n'+r'\caption{'+caption+'}'+r'\label{'+label+'}\n'+r'\centering\small\setlength{\tabcolsep}{4pt}'+'\n'+r'\begin{tabular}{l'+'r'*(len(headers)-1)+'}\n'+r'\toprule'+'\n'+' & '.join(headers)+r'\\\midrule'+'\n'
 text+='\n'.join(' & '.join(row)+r'\\' for row in rows)+'\n'+r'\bottomrule\end{tabular}\end{table}'+'\n';save(name,text,inputs)
def long_table(name,caption,headers,rows,inputs,spec=None):
 spec=spec or 'l'+'r'*(len(headers)-1)
 text=r'\begingroup\small\setlength{\tabcolsep}{3.5pt}'+'\n'+r'\begin{longtable}{'+spec+'}\n'+r'\caption{'+caption+'}'+r'\label{tab:'+name+'}'+r'\\'+'\n'+r'\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endfirsthead'+'\n'+r'\multicolumn{'+str(len(headers))+r'}{l}{\tablename\ \thetable\ (continued)}\\\toprule'+'\n'+' & '.join(headers)+r'\\\midrule\endhead'+'\n'+r'\midrule\multicolumn{'+str(len(headers))+r'}{r}{Continued on next page}\\\endfoot\bottomrule\endlastfoot'+'\n'
 text+='\n'.join(' & '.join(row)+r'\\' for row in rows)+'\n'+r'\end{longtable}\endgroup'+'\n';save(name,text,inputs)
path=D/'static-hqr/six-method-comparison.csv';rr=[r for r in read(path) if r['score']=='negative_energy' and r['comparison_budget']=='32' and r['comparison_delay']=='7' and r['view']=='all'];rr.sort(key=lambda r:['Fixed global','Fixed classwise','Static HQR','Rolling global','Dynamic HQR','Multi-anchor'].index(r['method']))
keys=['mean_daily_known_frr','period_background_macro_rejection','period_heldout_macro_rejection','mean_daily_known_correct_accepted'];rows=[[r['method'],*[pct(r[k]) for k in keys]] for r in rr]
main_table('revision-primary','Original-partition energy operating points (rates in \\%). Dynamic rules use32 requests/day and seven-day delay; all fixed rules and Static HQR request zero deployment labels. Static HQR is a post-confirmation ablation.','tab:revision-primary',['Method','FRR','BG macro','Held-out','Correct'],rows,[path])
path=D/'aggregate-detection-and-bootstrap.csv';rr=[r for r in read(path) if r['block_days']=='28'];labels={'background_macro_detection':'Background, class macro','background_micro_detection':'Background, flows','heldout_macro_detection':'Held-out, class macro','heldout_micro_detection':'Held-out, flows','all_unknown_macro_detection':'All unknown, class macro','all_unknown_micro_detection':'All unknown, flows'}
rows=[[labels[r['metric']],pct(r['rolling_global']),pct(r['hqr']),pct(r['difference']),f"[{pct(r['lower'])}, {pct(r['upper'])}]"] for r in rr if r['metric'] in labels]
main_table('revision-aggregates','Post-confirmation aggregation sensitivity for the original primary setting. Rates are percentages; differences and paired28-calendar-day95\\% intervals are percentage points.','tab:revision-aggregates',['Population/weight','Rolling','HQR',r'$\Delta$',r'95\% interval'],rows,[path])
path=D/'new-partition-v2/grid-means.csv';rr=[r for r in read(path) if r['score']=='negative_energy' and r['budget']=='32' and r['delay']=='7'];rows=[[r['method'],*[pct(r[k]) for k in ['known_frr','background_macro_detection','heldout_macro_detection','known_correct_accepted']]] for r in rr]
main_table('revision-new-partition','Supplementary fixed partition20261122, energy (rates in \\%). Dynamic rules use32 requests/day and seven-day delay; fixed rules use zero queries. The same network and sampled flow rows are reused.','tab:revision-new-partition',['Method','FRR','BG macro','Held-out','Correct'],rows,[path])
path=D/'class-specific-uncertainty.csv';rows=[]
for r in read(path):rows.append([esc(r['service']),{'known':'K','background':'B','heldout':'H'}[r['kind']],r['unique_n'],pct(r['rolling_global']),pct(r['hqr']),pct(r['difference']),f"[{pct(r['lower'])}, {pct(r['upper'])}]",r['valid_draws']])
long_table('revision-class-uncertainty','All53 true classes, original partition and primary setting. K/H/B denote known/held-out/background. Counts are unique flows. Rates are percentages; HQR--rolling differences and paired28-day intervals are pp. Valid denotes bootstrap draws with represented class support; this is not a simultaneous confidence family.',['Class','Type','$N$','Rolling','HQR',r'$\Delta$','Interval','Valid'],rows,[path],spec='llrrrrrr')
path=D/'leave-one-class-out.csv';rows=[[esc(r['excluded_service']),r['excluded_unique_n'],r['remaining_classes'],pct(r['rolling_global']),pct(r['hqr']),pct(r['difference']),f"[{pct(r['lower'])}, {pct(r['upper'])}]"] for r in read(path)]
long_table('revision-leave-one-out','Every within-population leave-one-class-out macro in the original primary setting. Removing a background group leaves two groups; removing a held-out service leaves19 services. Rates are percentages; differences and paired28-day intervals are pp.',['Excluded class','$N$','Remain','Rolling','HQR',r'$\Delta$','Interval'],rows,[path])
path=D/'aggregate-detection-and-bootstrap.csv';rows=[[esc(r['metric'].replace('_detection','').replace('_',' ')),r['block_days'],pct(r['rolling_global']),pct(r['hqr']),pct(r['difference']),f"[{pct(r['lower'])}, {pct(r['upper'])}]"] for r in read(path)]
long_table('revision-all-aggregate-blocks','Original primary population summaries under all three block lengths. Rates in percent; differences and intervals in pp. All estimates are exploratory supplements on the same traffic.',['Metric','Block','Rolling','HQR',r'$\Delta$','Interval'],rows,[path])
path=D/'risk-curves-v2/oracle-frr-matched.json';rr=json.loads(path.read_text());rows=[]
for r in sorted(rr,key=lambda r:(r['target_actual_frr'],r['method'])):
 assert r['status']=='INTERPOLATED';rows.append([r['method'],pct(r['target_actual_frr']),pct(r['left_nominal_alpha'])+'--'+pct(r['right_nominal_alpha']),*[pct(r[k]) for k in ['background_macro_detection','heldout_macro_detection','all_unknown_macro_detection','known_correct_accepted']]])
long_table('revision-oracle-points','Descriptive confirmation-oracle matched operating points. Adjacent nominal-risk segments are interpolated to actual confirmation FRR; no extrapolation or upper-envelope selection. These points are not deployable rules. All rates in percent.',['Method','FRR',r'Nominal $\alpha$','BG','Held','All macro','Correct'],rows,[path])
path=D/'development-risk-selection/development-selected-confirmation.csv';rows=[[r['method'],pct(r['target_known_frr']),pct(r['selected_nominal_alpha']),pct(r['achieved_development_frr']),pct(r['confirmation_known_frr']),pct(r['confirmation_background_macro_detection']),pct(r['confirmation_heldout_macro_detection'])] for r in read(path)]
long_table('revision-development-selection','Development-only nominal-risk selection followed by unchanged confirmation evaluation, all in percent. Selection minimizes development FRR distance from the target, with smaller-alpha tie break. This is separate from confirmation-oracle matching.',['Method','Target',r'$\alpha$','Dev FRR','Future FRR','BG','Held'],rows,[path])
path=D/'temporal-periods.csv';rr=read(path);rows=[]
for r in rr:
 if r['block_days'] not in ['7','28']:continue
 if r['period'].startswith('2024-') or r['period'].startswith('2025-'):
  if r['period'].count('-')==1 and len(r['period'])==7 and r['block_days']!='7':continue
 metric=r['metric'].replace('_detection','').replace('background','BG').replace('heldout','Held').replace('all_unknown','All').replace('known_correct_accepted','Correct').replace('known_frr','FRR').replace('_',' ')
 rows.append([r['period'],metric,r['block_days'],pct(r['rolling_global']),pct(r['hqr']),pct(r['difference']),f"[{pct(r['lower'])}, {pct(r['upper'])}]"])
long_table('revision-temporal-metrics','Monthly and requested multimonth period effects. Monthly intervals use7-day blocks; requested periods also show28-day blocks here, with all7/28/56-day records in source CSV. Rates in percent; differences/intervals in pp. Period comparisons are descriptive, without interaction tests.',['Period','Metric','Block','Rolling','HQR',r'$\Delta$','Interval'],rows,[path],spec='llrrrrr')
months=sorted({r['period'] for r in rr if len(r['period'])==7});rows=[]
for month in months:
 r=next(x for x in rr if x['period']==month);rows.append([month,r['calendar_days'],r['observed_dates'],r['known_n'],r['heldout_n'],r['background_n'],r['unknown_classes_present']])
long_table('revision-monthly-composition','Unique sampled-flow composition for the original service partition. Missing collection dates remain calendar gaps.',['Month','Calendar','Observed','Known','Held-out','Background','Classes'],rows,[path])
path=D/'sensitivity/grid-means.csv';rr=read(path)
for score in ['negative_energy','one_minus_msp','mahalanobis']:
 rows=[]
 for r in rr:
  if r['score']!=score:continue
  rows.append([esc(r['condition']),r['phase'],*[pct(r[k]) for k in ['known_frr','background_macro_detection','heldout_macro_detection','known_correct_accepted']]])
 long_table('revision-sensitivity-'+score,'Complete17-condition development-first sensitivity for '+esc(score)+'. Rates in percent,32 daily requests and seven-day delay. Each condition is fixed one-factor-at-a-time; no confirmation winner is selected.',['Condition','Phase','FRR','BG','Held','Correct'],rows,[path],spec='llrrrr')
path=D/'new-partition-v2/grid-means.csv';rr=read(path)
for score in ['negative_energy','one_minus_msp','mahalanobis']:
 rows=[[r['budget']+'/'+r['delay'],r['method'],*[pct(r[k]) for k in ['known_frr','background_macro_detection','heldout_macro_detection','known_correct_accepted']]] for r in rr if r['score']==score]
 long_table('revision-new-grid-'+score,'All new-partition20261122 resource settings for '+esc(score)+'. Rates in percent; B/D is requests per day/calendar delay for dynamic rules. Fixed global, fixed classwise and Static HQR use zero requests and are repeated as references, not independent samples.',['B/D','Method','FRR','BG','Held','Correct'],rows,[path],spec='llrrrr')
plan=V/'new-partition-plan.json';pp=json.loads(plan.read_text());inventory=json.loads((R/'outputs/inventory.json').read_text());names={int(v):k for k,v in inventory['apps'].items()};old=json.loads((R/'research/frozen/confirmation-v1.json').read_text());oldknown=set(old['known_codes']);rows=[[str(c),esc(names[c]),'K' if c in oldknown else 'H'] for c in pp['known_codes']]
long_table('revision-new-known-services','All30 known services in the supplementary partition20261122. Final column shows whether the same service was known(K) or held out(H) in the original partition.',['Code','Service','Original role'],rows,[plan],spec='rll')
path=D/'new-partition-v2/complete.json';rr=json.loads(path.read_text());rows=[[r['candidate']+' vs '+r['comparator'],esc(r['metric'].replace('_',' ')),str(r['block_days']),pct(r['difference']),f"[{pct(r['interval_95'][0])}, {pct(r['interval_95'][1])}]"] for r in rr['primary_contrasts']]
long_table('revision-new-blocks','New-partition primary paired contrasts under7/28/56-calendar-day blocks. Differences and intervals are pp, conditional on the same observed traffic; no independent-network replication is implied.',['Comparison','Metric','Block',r'$\Delta$','Interval'],rows,[path],spec='llrrr')
(T/'revision-source-mapping.json').write_text(json.dumps(dict(status='TABLES_GENERATED_FROM_VERIFIED_RESULTS_PENDING_RENDER_QA',tables=source_map),indent=2)+'\n');print(json.dumps(dict(tables=len(source_map),output=str(T))))
