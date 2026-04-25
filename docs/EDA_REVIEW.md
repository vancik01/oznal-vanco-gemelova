# EDA review — `analysis.Rmd`

Date: 2026-04-26
Method: every EDA chunk re-run from a clean R session against the live
2025 Oracle's Elixir CSV; plots saved to `/tmp/eda_review/*.png`,
numeric outputs in `run_log.txt` and `run_deep_log.txt`.

The current EDA is correct end-to-end — every claim in the prose is
backed by the data. What it lacks is the **diagnostic depth** a real
data scientist would expect: leakage checks, multicollinearity
quantification, calibration of the naive baseline, distributional
assumptions, league/patch heterogeneity, and multiple-testing.

---

## 1. Numeric claims — verification table

| Claim in Rmd | Value in Rmd | Recomputed | Status |
|---|---|---|---|
| Total raw rows | 120,636 | 120,636 | ✓ |
| Unique games | 10,053 | 10,053 | ✓ |
| Rows kept after `datacompleteness == "complete"` | 91.9 % | 91.87 % | ✓ |
| Games after pivot to one-row-per-game | implicit | 9,236 | ✓ (no partial games — every kept gameid has exactly 12 rows) |
| Blue WR | "~53 %" | 53.28 % (n=9,236) | ✓ |
| Blue WR 95 % CI | "excludes 50 %" | [52.26 %, 54.30 %] | ✓ |
| Side chi-sq p-value | "p ≪ 0.001" | 3.07 × 10⁻¹⁰ | ✓ |
| Naive "team ahead in gold @15 wins" accuracy | "~72.9 %" | 72.89 % (game-level), 72.89 % (team-level) | ✓ |
| Top correlation with `blue_win` | golddiffat15 | 0.520 | ✓ |
| Next two | xpdiffat15, csdiffat15 | 0.491 / 0.448 | ✓ |
| `blue_void_grubs` ↔ `red_void_grubs` | "strongly negative" | r = −0.624 | ✓ |
| `blue_goldat15` ↔ `blue_golddiffat15` | "multicollinear" | r = 0.798 | ✓ |
| First-tower chi-sq | "p ≪ 0.001" | χ² = 1467, p ≈ 5×10⁻³²¹ | ✓ |
| First-herald chi-sq | "p ≪ 0.001" | χ² = 960, p ≈ 9×10⁻²¹¹ | ✓ |
| First-dragon chi-sq | "p ≪ 0.001" | χ² = 275, p ≈ 1×10⁻⁶¹ | ✓ |
| First-blood chi-sq | "p ≪ 0.001" | χ² = 219, p ≈ 1×10⁻⁴⁹ | ✓ |
| 0 grubs WR | "well below 50 %" | 37.6 % (n=1,601) | ✓ |
| 6 grubs WR | "~70 %" | 69.3 % (n=857) | ✓ |
| Bot lane role correlation | highest | 0.374 (vs jng 0.346, mid 0.338, sup 0.316, top 0.294) | ✓ |
| `blue_goldat15` range | "~20,000 to ~36,000" | 20,419 – 35,540 | ✓ |
| `blue_killsat15` range | "0 – ~27" | 0 – 24 | ≈ (24 not 27) |
| Lead growth slope @10→@15 | "leads grow" | OLS slope 1.58, mean \|gd\| 1006 → 1969 | ✓ |
| Turret-plate diff correlation (line 596) | r = 0.413 | r = 0.4092 | ≈ (round to 0.41) |

**Two minor wording fixes** worth making:
- line 574: "blue_killsat15 on 0-~27" → actual max in the data is 24.
  Either change to "0 – 24" or keep "~27" only if you want to refer to
  the all-roles theoretical max.
- line 596: "r = 0.413 with result" → actual r = 0.409. Round to 0.41
  or update.

Everything else is exact.

---

## 2. Plot vs. prose — visual sanity check

| Plot | Prose claim | Verdict |
|---|---|---|
| `01_target.png` | Mild blue bias visible above the dashed line | matches |
| `02_corrplot.png` | Two block-clusters: raw stats (gold/xp/cs at 10 & 15) inter-correlate; mirrors (blue vs red void grubs / turret plates) anti-correlate | matches |
| `03_corbar.png` | Differentials at the top, first-blood at the bottom | matches |
| `04_golddiff_density.png` | Two centered, roughly symmetric bell curves shifted ~±1.5k around 0 | matches; **note** the heavy overlap zone ±2k where ~33 % of games live |
| `05_grubs.png` | Monotonic with a small dip at 4 grubs | matches; the dip is small and explained by sample size |
| `06_roles.png` | Bot leads, top trails | matches |
| `07_league_wr.png` *(new)* | Range 44 – 60 % across leagues — shows global 53 % is an average over very different competitive scenes | not currently in Rmd |

---

## 3. Issues found

### 3.1 Magnitude order vs. prose

Line 413: *"first blood and first dragon carry less signal"* — true for
correlation magnitude, but **first dragon's chi-sq (275) is slightly
larger than first blood's (219)**. The prose pairs them, and that's
fair, but if you want to be precise: first blood swings WR from 45 % to
61 %, first dragon swings it from 47 % to 64 %. First dragon is
actually a marginally bigger swing in absolute pp.

### 3.2 Grub "wobble" attribution

Line 445: *"wobbles ... come from low sample sizes — teams that end up
with exactly 4 grubs are rare"*. n at 4 grubs = 503; n at 5 = 477 (also
rare); n at 3 = 3,183 and at 6 = 857. So the 3-3 split is overwhelmingly
modal, then 6-0 / 0-6, then everything else. The wobble is real and the
sample-size explanation is correct, but the wording could note that the
mode at 3 reflects a "split-then-leave" dynamic, not a "winner-take-all"
one as the current prose implies.

### 3.3 Patch range correctness

The dataset is **patch 15.01 – 15.24, all major version 15** (verified).
So the patch hedge added in the previous commit ("13:45 on early-2025
patches; moved to 14:45 mid-season") is appropriate — both regimes are
in the data.

---

## 4. Missing diagnostics — what a data scientist would add

### 4.1 Multicollinearity, quantified (VIF)

The Rmd mentions multicollinearity qualitatively. Computed VIFs:

```
blue_golddiffat15  19.14  ← severe
blue_goldat15      18.50  ← severe
blue_killsat15     12.86
blue_deathsat15    10.60
blue_csat15         9.40
blue_csdiffat15     4.79
blue_assistsat15    4.75
blue_xpdiffat15     4.38
blue_xpat15         4.32
blue_turretplates   2.43
red_turretplates    2.30
red_void_grubs      1.73
blue_void_grubs     1.67
all `first*` flags  ≤ 1.42
```

**Recommendation — placement matters.** VIF is a *diagnostic*, not a
feature-selection step. Two defensible places, given the S1/S3
narrative arc:

- **Option A (EDA):** include the VIF table in the EDA, but frame it
  explicitly as motivation for the Scenario 3 regularised models —
  *not* as a reason to drop features in S1. The S1 design is to
  compare LR, RF, NB and KNN on the same feature set; dropping
  collinear features in EDA would break that comparison.
- **Option B (preferred):** put the VIF table **inside Scenario 1,
  next to the Logistic Regression fit**, as the explanation for the
  rank-deficient warning LR produces. This ties cause (VIF = 19) to
  symptom (warning) to fix (S3 LASSO/EN) in a single narrative
  thread, and keeps the EDA descriptive while the modelling chapters
  stay diagnostic.

The thing to avoid is a VIF table in EDA without framing — a reader
will assume features are about to be dropped, then get confused when
S1 keeps them all.

### 4.2 Calibration of the naive baseline

The 72.9 % baseline is **dominated by easy games**. Per-bin accuracy:

```
|gold diff @15|     acc    n
[0, 500]           53.1%  1,601   ← coin flip
(500, 1k]          61.0%  1,511
(1k, 2k]           71.0%  2,532
(2k, 3k]           80.3%  1,554
(3k, 5k]           92.2%  1,496
(5k, +inf]         98.3%    542
```

**Recommendation:** add this table or a calibration plot. It reframes
the modelling task — ML's value is concentrated in the 33 % of games
with |gd| < 1k, not in the runaway games where the rule is already 92 %+.
This is a much stronger framing for the modelling chapter than "we beat
73 %".

### 4.3 League heterogeneity

Blue WR per league (n ≥ 50) ranges from **44.0 % (DCup) to 59.8 %
(NLC)** — a 16 pp spread. The global chi-sq across all 39 leagues is
only p = 0.236 (not significant after pooling), but individual smaller
leagues differ a lot from the global rate. None of this is currently
discussed.

**Recommendation:**
1. Add a "Blue WR by league" plot (top 20 by n) — already saved as
   `07_league_wr.png` in the review folder.
2. Either include `league` as a feature (one-hot) or stratify the
   train/test split by league so the model isn't optimising for the
   league mix.

### 4.4 Single-feature AUC scan (leakage / dominance check)

```
blue_golddiffat15  AUC = 0.810
blue_xpdiffat15    AUC = 0.790
blue_csdiffat15    AUC = 0.762
blue_golddiffat10  AUC = 0.749
blue_goldat15      AUC = 0.748
```

No feature exceeds 0.85 → no leakage red flag. But `golddiffat15` alone
already reaches AUC 0.81; that puts a **ceiling on how much extra ML
can deliver** and should be reported. `red_*` mirror columns appear
with identical AUC, confirming the strict mirror property — useful as a
sanity check.

**Recommendation:** add a single-feature AUC table; it is the cleanest
way to show what each feature alone is worth.

### 4.5 Naive Bayes assumption check (Gaussian)

Shapiro-Wilk p-values (subsampled to n=5,000 where needed):

```
blue_goldat15      p = 3 × 10⁻³⁴   skew = +0.86
blue_xpat15        p = 0.018       skew = -0.08  ← closest to normal
blue_csat15        p = 2 × 10⁻²¹   skew = -0.46
blue_killsat15     p = 9 × 10⁻⁴³   skew = +1.05  ← strongly skewed
blue_void_grubs    p = 2 × 10⁻⁴⁷   skew = +0.34  (count, not continuous)
blue_golddiffat15  p = 6 × 10⁻¹⁶   skew =  0.03  ← symmetric, but heavy tails
```

**Every** feature fails Shapiro. Counts (kills, grubs) are not Gaussian
at all. The Rmd justifies skipping standardisation for Naive Bayes
("Tree-based models … and Naive Bayes do not need it") but **does not
mention that GaussianNB's distributional assumption is violated**.

**Recommendation:** add one paragraph to the scaling-check section: NB
will be biased on count features; either (a) discretise the count
features for Categorical/Multinomial NB, (b) log-transform skewed ones,
or (c) acknowledge the limitation when interpreting NB results.

### 4.6 Outlier audit

Z-score > |3| counts: 87 (blue_goldat15), 102 (blue_killsat15), 77
(blue_golddiffat15). Top-5 highest blue_goldat15 are all from smaller
leagues (LPLOL, AL, EBL, LJL). They look real (blowout games), not data
errors — but the Rmd never says so.

**Recommendation:** add a one-line note: "Z>|3| outliers ~1 % of games;
inspected — all from smaller leagues, no data quality issues."

### 4.7 Interaction check (for free, since you discuss them implicitly)

Tested gold-diff × first-tower interaction:
- gold-only AIC 9,695
- tower-only AIC 11,261
- additive AIC 9,586
- interaction AIC 9,586 (LRT p = 0.155)

→ The additive model is sufficient; no interaction effect. This
**supports** the choice of plain LR / no engineered interaction
features. Currently the Rmd implies interactions matter ("RF should
handle … interactions between objectives and economy") without testing
the hypothesis.

**Recommendation:** add the LRT result, or drop the implication.

### 4.8 Multiple-testing adjustment

4 first-objective chi-sq tests. After Benjamini-Hochberg:
all four still significant by orders of magnitude. Conclusion
unchanged, but explicit BH-adjusted p-values are good hygiene.

### 4.9 Effect sizes for binary objectives

Currently only chi-sq + bar plot. Adding an **odds-ratio table with
95 % CI** is more informative than chi-sq for a reader:

| Objective | OR (Blue secured / Red secured) | 95 % CI |
|---|---|---|
| First tower | ~5.4 | very high |
| First herald | ~4.0 | very high |
| First dragon | ~2.1 | high |
| First blood | ~1.9 | moderate |

(Easy to compute with `epitools::oddsratio`.)

### 4.10 EDA done on the full dataset, not the train split

Best practice: EDA on the training set only, so test-set distributions
remain unseen. Currently every correlation/chi-sq uses all 9,236 games
including what becomes the test fold.

**Recommendation:** move the train/test split *above* the EDA, or at
least note that the EDA correlations are descriptive of the entire
dataset and used only to choose features, not validate them.

---

## 5. Suggested additions to the EDA Summary

Add these bullets (between current points 11 and 12):

> 12. **Naive baseline accuracy is a function of lead size**: from
>     53 % when |gd| < 500 to 98 % when |gd| > 5k. The ML models'
>     value is concentrated in the close-game band.
> 13. **VIF analysis** flags blue_goldat15 (18.5) and
>     blue_golddiffat15 (19.1) as severely collinear — kept in S1 by
>     design (so all four models see the same input) and addressed
>     in S3 via LASSO / Elastic Net.
> 14. **League distribution is heterogeneous** (Blue WR 44–60 % across
>     leagues, n ≥ 50). Either include league as a feature or
>     stratify the split.
> 15. **Naive Bayes Gaussian assumption is violated** for every feature
>     (Shapiro p ≪ 0.05; kills skew = +1.05). Interpret NB scores
>     with this caveat.

---

## 6. Files generated

In `/tmp/eda_review/`:

- `run_eda.R` — reproduction of every EDA chunk
- `run_log.txt` — text outputs from above
- `run_deep.R` — additional diagnostic checks (this report's §4)
- `run_deep_log.txt` — text outputs from above
- `01_target.png` … `08_patch_wr.png` — saved plots

Both R scripts are self-contained and can be re-run:

```bash
cd /Users/martinvanco/Documents/FIIT/4_ROC/OZNAL/projekt
Rscript /tmp/eda_review/run_eda.R
Rscript /tmp/eda_review/run_deep.R
```
