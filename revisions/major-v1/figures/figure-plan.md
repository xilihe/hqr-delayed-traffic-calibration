# Revision result figures

PROJECT_REUSE of the existing PAA result style and its verified reference-selection record at research/figures/reference-selection.md. Preparation remains delegated; no claim is made that the author separately inspected or approved each new figure. No source pixels are reused.

All panels are RESULT / JOURNAL_RESULT. Python/Matplotlib Agg; 130 mm vector PDF and editable-text SVG; 300 dpi PNG preview; sans-serif 8 pt labels; white background; stable method colours, line styles and symbols; no decorative assets. The new Static HQR uses purple squares and a dashed line to distinguish it from dynamic HQR.

- Risk grid: three aligned panels for background, held-out and all-unknown period macros against actual known FRR. Six methods, every declared nominal-alpha point, and larger symbols at original nominal 5%. These are operating-point grids, not a single-model ROC. Fixed thresholds use no deployment labels. No uncertainty band is invented. Oracle matched values remain explicitly separate descriptive interpolation results.
- Class robustness: all 23 unknown-class effects and all 23 within-population leave-one-out effects, with paired 28-day intervals and class support displayed. Include negative and positive values; no favourable-class selection.
- Time stability: eight calendar months, paired monthly differences for background and held-out detection and their 7-day block intervals, plus flow composition. Do not interpret differences between month point estimates as a formal interaction test.
- Mechanism: historical C tail dispersion and monthly mixture/conditional-rate decomposition, clearly labelled fixed historical thresholds and flow-weighted diagnostic risk. Association and algebraic decomposition do not imply causality.
- Sensitivity: complete fixed OFAT grid in machine-readable tables; selected display groups follow parameter families, not best observed outcomes. No confirmation winner is selected.
- New split: render only after actual training, scoring and independent evaluation finish. Identify service split, model and audit replications; never label it independent-network validation.

Every rendered figure must retain input paths/hashes, transformations and a matching caption, then undergo actual final-size/grayscale inspection before insertion in the revised manuscript.
