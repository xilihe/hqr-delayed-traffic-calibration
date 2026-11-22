"""Build the result prose and abstract from the audited point/interval sources."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'outputs/confirmation-analysis-v1/analysis.json').read_text())
p=json.loads((ROOT/'research/frozen/confirmation-v1.json').read_text())
with (ROOT/'outputs/confirmation-analysis-v1/operating-grid.csv').open() as f:grid=list(csv.DictReader(f))
raw=a['summaries']['raw:rolling_global|all'];hqr=a['summaries']['shift:rolling_global|all'];multi=a['summaries']['reference:rolling_global|all']

def rate(x):return f'{x*100:.2f}'
def contrast(metric,comparator='raw:rolling_global',view='all',block=28):return next(r for r in a['contrasts'] if r['metric']==metric and r['comparator']==comparator and r['view']==view and r['block_days']==block)
def interval(r):return f"{rate(r['interval_95'][0])} to {rate(r['interval_95'][1])}"
def group(v,score,b,d,field):return statistics.mean(float(r[field]) for r in grid if r['variant']==v and r['method']=='rolling_global' and r['score']==score and int(r['budget'])==b and int(r['delay'])==d and r['view']=='all')
def class_rate(method,label):return next(r['mean_rejection'] for r in a['class_detection'] if r['method']==method and r['label']==label and r['view']=='all')
bg=contrast('background_macro_detection');held=contrast('heldout_macro_detection');frr=contrast('known_frr');utility=contrast('known_correct_accepted')
means=[]
for score in p['scores']:
 for b in p['budgets']:
  for d in p['delays']:
   means.append(group('shift',score,b,d,'period_background_macro_rejection')-group('raw',score,b,d,'period_background_macro_rejection'))
assert len(means)==18 and min(means)>0
model=[]
for seed in [42,43,44]:
 x={r['method']:r for r in a['model_sensitivity'] if r['model_seed']==seed and r['view']=='all'}
 model.append(x['shift:rolling_global']['background_macro_detection']-x['raw:rolling_global']['background_macro_detection'])
text=rf'''\section{{Results}}\label{{sec:results}}
\subsection{{Frozen temporal confirmation}}
The confirmation evaluated 1,417,414 sampled flows across 238 observed dates, including 764,977 known-service flows, 144,912 held-out-service flows and 507,525 background flows (Table~\ref{{tab:cohorts}}). Every source-model seed used the same sampled flow identities. Figure~\ref{{fig:cohort}} shows the changing traffic composition and the known-flow classification accuracy before rejection. Model weights, historical mappings and source preprocessing remained fixed throughout this period.

\input{{tables/cohort-counts}}
\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figure1-cohort.pdf}}
\caption{{Confirmation cohort. (a) Observed-date proportions of known services, held-out services and the three background groups. White gaps mark dates absent from the released dataset. (b) Known-flow classification accuracy before rejection, displayed over trailing 28-calendar-day windows with at least seven observed dates. The line is the mean of the three source-model seeds; shading is their range. Display smoothing does not enter the frozen statistical estimates.}}
\label{{fig:cohort}}
\end{{figure}}

\subsection{{Background rejection at the primary operating point}}
With energy scores, 32 requested labels per day and a seven-day delay, HQR increased background macro detection from {rate(raw['background_macro_detection'])}\% to {rate(hqr['background_macro_detection'])}\% (Table~\ref{{tab:primary}}). The paired difference was {rate(bg['difference'])} percentage points (pp), with a 95\% paired block-bootstrap interval of {interval(bg)} pp. Mean daily known false rejection was {rate(raw['known_frr'])}\% for raw rolling-global calibration and {rate(hqr['known_frr'])}\% for HQR. Its paired change was {rate(frr['difference'])} pp, with an interval of {frr['interval_95'][0]*100:.3f} to {frr['interval_95'][1]*100:.3f} pp. The point difference is within the 0.5-pp operational reference, while its upper interval endpoint is 0.501 pp. These are achieved operating points for a nominal 5\% target; Table~\ref{{tab:contrasts}} reports the paired contrasts.

\input{{tables/primary-results}}
\input{{tables/paired-contrasts}}

The detection change differed between unknown populations. Held-out-service macro detection was {rate(hqr['heldout_macro_detection'])}\% for HQR and {rate(raw['heldout_macro_detection'])}\% for raw rolling-global calibration, a difference of {rate(held['difference'])} pp (95\% interval, {interval(held)} pp). Correct-and-accepted known predictions changed from {rate(raw['known_correct_accepted'])}\% to {rate(hqr['known_correct_accepted'])}\%, a difference of {rate(utility['difference'])} pp (interval, {interval(utility)} pp). The primary result identifies a background-oriented operating trade-off.

The other calibration rules illustrate why detection must be read together with known false rejection. Fixed classwise calibration reached {rate(a['summaries']['raw:fixed_classwise|all']['background_macro_detection'])}\% background detection at {rate(a['summaries']['raw:fixed_classwise|all']['known_frr'])}\% known false rejection. Rolling classwise calibration gave {rate(a['summaries']['raw:rolling_classwise|all']['background_macro_detection'])}\% and {rate(a['summaries']['raw:rolling_classwise|all']['known_frr'])}\%, respectively. Delayed ACI remained close to raw rolling-global calibration, with {rate(a['summaries']['raw:delayed_aci_global|all']['background_macro_detection'])}\% background detection and {rate(a['summaries']['raw:delayed_aci_global|all']['known_frr'])}\% known false rejection. Figure~\ref{{fig:primary}} shows how these operating rates varied over time.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figure2-primary.pdf}}
\caption{{Primary temporal comparison: energy scores, 32 labels per day and a seven-day delay. Panels show (a) known false rejection, (b) background macro detection and (c) held-out-service macro detection. Curves use trailing 28-calendar-day display windows with at least seven observed dates. Unknown-class counts are aggregated within each window before the class macro is calculated. Lines average the nine fixed model/audit cells; shading gives the range of the three model-specific audit means, not a confidence interval. The dashed horizontal line is the nominal 5\% known-rejection target.}}
\label{{fig:primary}}
\end{{figure}}

\subsection{{Budget, delay and base-score dependence}}
The background benefit appeared at the smallest tested audit budget. For seven-day-delayed energy calibration, HQR background macro detection was {rate(group('shift','negative_energy',32,7,'period_background_macro_rejection'))}\% at 32 daily requests and {rate(group('shift','negative_energy',512,7,'period_background_macro_rejection'))}\% at 512. Known false rejection was {rate(group('shift','negative_energy',32,7,'mean_daily_known_frr'))}\% and {rate(group('shift','negative_energy',512,7,'mean_daily_known_frr'))}\%, respectively. The period means also changed little between one- and seven-day delays in these settings (Figure~\ref{{fig:budget}}). This comparison concerns the tested collection and calendar window; both information age and available audit support change with delay.

Across the 18 frozen score/budget/delay cells, HQR had higher mean background macro detection than raw rolling-global calibration. The increases ranged from {rate(min(means))} to {rate(max(means))} pp. At the primary budget and delay, the increase was {rate(group('shift','one_minus_msp',32,7,'period_background_macro_rejection')-group('raw','one_minus_msp',32,7,'period_background_macro_rejection'))} pp for the maximum-softmax complement and {rate(group('shift','mahalanobis',32,7,'period_background_macro_rejection')-group('raw','mahalanobis',32,7,'period_background_macro_rejection'))} pp for Mahalanobis distance. These grid summaries are descriptive sensitivity analyses. Held-out-service effects depended on the base score: maximum-softmax-complement detection changed from {rate(group('raw','one_minus_msp',32,7,'period_heldout_macro_rejection'))}\% to {rate(group('shift','one_minus_msp',32,7,'period_heldout_macro_rejection'))}\%, whereas the energy result exhibited the trade-off reported above. Supplementary Tables~S2--S4 retain every frozen grid cell.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figure3-budget-delay.pdf}}
\caption{{Energy calibration across the frozen audit budgets and delays. Top panels show period-level background macro detection; bottom panels show mean daily known false rejection. Columns use one- and seven-day feedback delays. Points average the common audit streams and source-model seeds. Whiskers show the range of the three source-model means, each averaged over its three audit streams. Fixed classwise calibration uses zero deployment queries and is repeated as a static reference.}}
\label{{fig:budget}}
\end{{figure}}

\subsection{{Historical mapping, population and sensitivity analyses}}
The single historical tail offset retained the benefit of the multi-anchor transformation. At the primary setting, HQR gave {rate(hqr['background_macro_detection'])}\% background macro detection, compared with {rate(multi['background_macro_detection'])}\% for the multi-anchor control. The paired difference was {rate(contrast('background_macro_detection','reference:rolling_global')['difference'])} pp (95\% interval, {interval(contrast('background_macro_detection','reference:rolling_global'))} pp). HQR also had {rate(contrast('heldout_macro_detection','reference:rolling_global')['difference'])} pp higher held-out-service detection. The added anchors did not provide a mean detection advantage in this frozen primary comparison.

The background change was concentrated in particular groups (Figure~\ref{{fig:sensitivity}}a). Detection of default-background changed from {rate(class_rate('raw:rolling_global',0))}\% to {rate(class_rate('shift:rolling_global',0))}\%; facebook-background changed from {rate(class_rate('raw:rolling_global',52))}\% to {rate(class_rate('shift:rolling_global',52))}\%. Google-background changed from {rate(class_rate('raw:rolling_global',51))}\% to {rate(class_rate('shift:rolling_global',51))}\%. The macro gain summarizes these heterogeneous changes with equal weight for each group.

As a flow-weighted secondary summary, background detection increased from {rate(group('raw','negative_energy',32,7,'period_background_micro_rejection'))}\% to {rate(group('shift','negative_energy',32,7,'period_background_micro_rejection'))}\%. True-service macro known rejection was {rate(group('raw','negative_energy',32,7,'period_known_macro_rejection'))}\% and {rate(group('shift','negative_energy',32,7,'period_known_macro_rejection'))}\%, respectively. An exploratory true-service breakdown gave a maximum of {rate(max(r['mean_rejection'] for r in a['class_detection'] if r['method']=='raw:rolling_global' and r['view']=='all' and r['kind']=='known'))}\% among the 30 service-specific mean rejection rates for raw rolling-global calibration and {rate(max(r['mean_rejection'] for r in a['class_detection'] if r['method']=='shift:rolling_global' and r['view']=='all' and r['kind']=='known'))}\% for HQR. Supplementary Table~S7 reports every class, including classes whose rejection increased.

The three source-model-specific background gains ranged from {rate(min(model))} to {rate(max(model))} pp. Excluding the 8,216 confirmation inputs equal to fitting or historical-calibration inputs retained a background difference of {rate(contrast('background_macro_detection',view='input_disjoint')['difference'])} pp; the corresponding held-out-service difference was {rate(contrast('heldout_macro_detection',view='input_disjoint')['difference'])} pp. Using seven- and 56-day bootstrap blocks gave background-difference intervals of {interval(contrast('background_macro_detection',block=7))} pp and {interval(contrast('background_macro_detection',block=56))} pp. Supplementary Tables~S5--S6 show the model and block-length results. These checks preserve the background effect while keeping its population and score boundaries visible.

\begin{{figure}}[tbp]
\centering
\includegraphics[width=\linewidth]{{figure4-mechanism-sensitivity.pdf}}
\caption{{Population and mapping sensitivity. (a) Primary rejection rates for each official background group. (b) HQR minus rolling-global background detection and known false rejection at 32 labels per day and a seven-day delay, for each base score; whiskers are ranges across source-model audit means. (c) Primary HQR minus rolling-global detection differences for all inputs and the input-disjoint sensitivity. (d) Corresponding HQR minus multi-anchor differences. Intervals in (c,d) are paired 28-calendar-day bootstrap 95\% intervals. Filled/open symbols distinguish the two input views. All differences use percentage points (pp).}}
\label{{fig:sensitivity}}
\end{{figure}}
'''
(ROOT/'manuscript/sections/results.tex').write_text(text)
abstract=f'''Recent labels are costly for maintaining rejection rules in encrypted traffic classification, while historical calibration contains service-specific information that can outlast model fitting. We study historical quantile recentering (HQR), which subtracts a fixed historical tail quantile for the predicted service and updates one shared threshold from delayed audit labels. Historical mapping and initial risk calibration use disjoint flows; every requested unknown label consumes the audit budget. A frozen evaluation on CESNET-QUICEXT-25-XS covers 238 dates and 1,417,414 flows over October 2024--May 2025, with three source-model seeds and three common audit streams. At 32 daily labels and a seven-day delay, energy-based HQR increased background macro detection from {rate(raw['background_macro_detection'])}\\% to {rate(hqr['background_macro_detection'])}\\%, a gain of {rate(bg['difference'])} percentage points (95\\% paired block-bootstrap interval, {interval(bg)}). Known false rejection changed from {rate(raw['known_frr'])}\\% to {rate(hqr['known_frr'])}\\%; held-out-service detection changed from {rate(raw['heldout_macro_detection'])}\\% to {rate(hqr['heldout_macro_detection'])}\\%. Background gains appeared across all 18 tested score/budget/delay cells, and a multi-anchor transformation offered no mean detection advantage in the primary comparison. The results support reusing historical class quantiles for background-oriented rejection under small delayed audits, while showing that benefits depend on the unknown population and base score.'''
(ROOT/'manuscript/sections/abstract.tex').write_text(abstract+'\n')
(ROOT/'research/verification/results-prose-sources.json').write_text(json.dumps({'status':'PROSE_GENERATED_FROM_VERIFIED_CONFIRMATION','primary':p['primary'],'sources':['outputs/confirmation-analysis-v1/analysis.json','outputs/confirmation-analysis-v1/operating-grid.csv'],'primary_background_difference':bg,'primary_heldout_difference':held,'primary_known_frr_difference':frr,'all_grid_background_difference_range':[min(means),max(means)],'abstract_word_count_approximate':len(abstract.split())},indent=2)+'\n')
print('Wrote results and abstract; abstract words:',len(abstract.split()))
