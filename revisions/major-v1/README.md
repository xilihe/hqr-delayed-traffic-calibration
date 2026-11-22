# Supplementary revision: historical quantile recentering

These additions follow inspection of the original frozen confirmation results. They are post-confirmation supplementary/exploratory analyses. The original study, internal protocol and published results remain available at commit `e14d4847f56bc00524ee729952fb2d0e509d6e95`; the original freeze does not cover these additions.

The seven experiments separate historical service offsets from recent threshold updates, audit class concentration and aggregation, compare achieved operating risk, measure temporal effects and conditional drift, retain a fixed sensitivity grid, and evaluate a second service partition. The second partition was fixed at seed 20261122 before its new training and outcome inspection. No favorable partition replacement was performed. Both partitions use the same network and sampled flows.

At the original energy/32-request/7-day setting, background macro detection changes from 16.90% to 20.35%, while held-out macro changes from 17.42% to 15.41%. Static HQR reaches 24.32% background detection at 7.45% known FRR, compared with Dynamic HQR's 20.35% at 5.19%. Their descriptive matched-5%-FRR background points are 20.04% and 19.98%. The second partition has a +0.90 pp background contrast with a 28-day interval crossing zero, and lower held-out detection. The full tables retain the adverse classes, alternative aggregations and all fixed conditions.

## Recreate tables and figures from retained results

From the repository root, in an environment with NumPy and Matplotlib:

```bash
python revisions/major-v1/scripts/build_revision_tables.py
python revisions/major-v1/scripts/plot_risk_curves.py
python revisions/major-v1/scripts/plot_class_time.py
python revisions/major-v1/scripts/plot_mechanism.py
```

This route regenerates presentation from retained observations; it does not repeat fitting or score inference. Tables are written to `revisions/major-v1/manuscript/tables`. Plot PDFs, editable SVGs and PNG previews are written to `revisions/major-v1/figures`. Source mappings record the input hashes and transformations. The paper PDFs and editable manuscript are delivered separately.

## Recompute from the official traffic data

First complete the original full-reproduction route in a new directory, using `scripts/run_reproduction.py` and the exact official XS dataset described in the root README. This regenerates the original models, historical and future score caches, result counts and a valid internal seal. Do not point the reproduction runner at the published checkout or the author's completed working directory.

Then inspect and run the supplementary stage plan, using the same new directory:

```bash
python revisions/major-v1/scripts/run_revision_reproduction.py --work-dir /absolute/path/to/new-original-rerun --plan
python revisions/major-v1/scripts/run_revision_reproduction.py --work-dir /absolute/path/to/new-original-rerun --stage initialize
python revisions/major-v1/scripts/run_revision_reproduction.py --work-dir /absolute/path/to/new-original-rerun --stage original-supplements
python revisions/major-v1/scripts/run_revision_reproduction.py --work-dir /absolute/path/to/new-original-rerun --stage new-partition-fit
python revisions/major-v1/scripts/run_revision_reproduction.py --work-dir /absolute/path/to/new-original-rerun --stage new-partition-evaluation
python revisions/major-v1/scripts/run_revision_reproduction.py --work-dir /absolute/path/to/new-original-rerun --stage presentation
```

Allocate a CUDA environment for `new-partition-fit`, which trains and scores three source-only models for the new service set using the original architecture and 50-epoch schedule. The remaining scientific replay/statistics stages are CPU computations; presentation also needs Matplotlib. The experiment reference environment is Python 3.11, NumPy 1.26.4, PyTables 3.10.2 and PyTorch 2.5.1 with CUDA 12.4. Final publication plotting was also executed in a separate Python 3.12/NumPy 2.5.4/Matplotlib 3.11.2 environment. No SSH account, lab scheduler or external API is required by this public runner.

The stage runner records commands and terminal status and refuses to overwrite existing scientific stage receipts. After a failure, inspect its recorded command and partial outputs before retrying. The supplementary runner reuses the three original-partition models and fits only the three necessary new-partition models. Exact model bytes may require matching the recorded hardware/software environment; a fresh rerun creates its own hash-consistent seal rather than editing the published seal.

## Analysis map

- `analyse_saved_counts.py`: independently reconstructs original metrics, every class, all leave-one-out analyses, all-unknown aggregation, sample-size association and monthly/period effects.
- `static_hqr.py`: fixed historical H mapping and independent C threshold, zero deployment queries, all scores/resource references and input views; separate Boolean-count verification.
- `risk_curves.py`: all declared risk-grid points and regression checks against original nominal-5% class counts.
- `bootstrap_risk_matching.py`: descriptive confirmation-oracle interpolation with paired calendar-block intervals; no extrapolation or upper envelope.
- `development_risk_selection.py`: selects nominal risk using development known FRR only, writes selection first, then reads confirmation results.
- `mechanism.py` and `class_attribution.py`: same-C tail diagnostics, predicted-group mixtures, exact fixed-threshold mixture/conditional decomposition and actual decision-switch contributions. These are descriptive, not causal effects.
- `sensitivity.py`: all 17 declared conditions, development phase before confirmation; no confirmation winner.
- `new_partition_pipeline.py`, `score_new_partition.py`, `evaluate_new_partition.py`: new fixed split training, inference and six-method resource-grid evaluation.
- `independent_array_audit.py` and `audit_new_partition_aggregates.py`: separate threshold/count reconstruction and independent standard-library aggregate checks.

## Statistical meaning and retained files

Known FRR and correct-and-accepted rates average observed dates. Period macros aggregate each true class's counts over the period before averaging classes equally. Micro rates weight flows. Three model fits and three audit streams share the traffic and do not constitute nine independent datasets. Paired moving blocks use calendar days, with common weights across methods and replications; missing dates remain gaps. The 28-day reference uses 2,000 resamples, with 7/56-day sensitivity. Monthly exploratory intervals use 7-day blocks. No simultaneous family-wise superiority claim is made. Original known-FRR noninferiority at +0.5 pp is not established, and the original held-out interval crosses zero with 56-day blocks.

The public release retains plain CSV/JSON summaries, scientific source and configurations. Source-model checkpoints and row maps for the new partition are provided as fitting references. Large derived score caches and risk/count NPZ intermediates are regenerated by the full route rather than uploaded. In the local delivery, those count NPZ files are retained with the experiment outputs. The data remain at the official CESNET/Zenodo source and are not redistributed. Engineering fixtures are labeled as tests and never supply result values.
