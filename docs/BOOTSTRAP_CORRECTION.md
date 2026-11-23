# Fixed observed-day scale for exceedance intervals

The original2000 paired28-calendar-day draws sample243calendar positions with5missingdates. Effective observed-date weights vary from220to243, so raw weighted exceedance counts do not represent a fixed238-day count. The corrected release reports proportions directly and defines equivalent days per draw as238times that draw's exceedance proportion, before taking percentiles. No clipping is applied. S24/S26/S28/S30count intervals change; all observed points, fractions, MAE, RMSE andp90 remain unchanged.

For example, second-partition nominal Fixed classwise at>5.5% keeps point231.00 but its equivalent-day95% interval is[224.89,236.04], replacing the wrongly labelled[217.67,240.67]. Static HQR gives230.67[221.93,236.70]. This changes the derived-count reporting, not classifier decisions or the frozen primary study.

The public offline driver redoes the statistical computations from retained counts and records every executed command. See the versioned README for exact inputs, outputs and limits.
