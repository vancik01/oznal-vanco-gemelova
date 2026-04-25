# Fact-check report — `analysis.Rmd` citations

Date: 2026-04-26

## Scope

Verification of every external citation and game-mechanic claim in the
intro and Related Work sections of `analysis.Rmd`. Each source was
located, downloaded where possible, and checked against the wording or
numbers we use.

## Local PDF copies

Stored in `sources/`:

- `wickham_2014_tidy_data.pdf`
- `lin_2016_stanford_cs229.pdf`
- `spaargaren_2022_tilburg_msc.pdf`
- `junior_campelo_2023_arxiv_2309.02449.pdf`

Two papers could not be downloaded:

- **Chowdhury et al. 2025** (MDPI *Applied Sciences* 15(10):5241) — MDPI
  serves an HTML stub to every UA tried; numbers verified via
  search/abstract.
- **Lafrance & Grewal 2026** (Elsevier *Entertainment Computing* vol. 57,
  S1875952126000108) — ScienceDirect 403/HTML stub on every attempt;
  numbers verified via search/abstract. Authors confirmed by name as
  **Artin Lafrance and Ratvinder Grewal**.

If local copies are needed, both are available from a campus IP /
institutional mirror.

## Verified — left as-is

| Citation | Where in Rmd | Status |
|---|---|---|
| Wickham 2014, *JSS* 59(10), tidy-data rule | line 136 | ✓ exact match |
| Lin 2016, Stanford CS229 (single-row schema) | line 136 | ✓ confirmed |
| Spaargaren 2022 — LR 77.99 / RF 77.26 / SVM 76.99 / NB 76.82 / kNN 74.83 | lines 62–66 | ✓ all five reproduce Appendix E |
| Lafrance & Grewal 2026 — XGBoost 77.1 % Bronze, 76.9 % Diamond @15 min | lines 67–68 | ✓ confirmed |
| Tsang 2025 — `golddiffat15` coef >1.0 (2020-23) → 0.83 (2025); first blood flips negative | line 77 | ✓ confirmed |
| Chowdhury 2025 — 76.8 % peak with pre+in-game features | line 79 | ✓ confirmed |
| Atakhan & Feats of Strength introduced 2025 | line 79 | ✓ confirmed (planned removal in 2026) |
| Turret plate armor falls at 14:00 | lines 42, 583, 591 | ✓ still current |

## Issues found and fixed

### 1. Junior & Campelo 2023 — wrong numbers

**Before** (lines 69–71):

| Junior & Campelo | LightGBM | ~77.9% |
| Junior & Campelo | Random Forest | ~77.8% |
| Junior & Campelo | Logistic Regression | ~77.2% |

Two errors:

1. The values shown were taken at **60 % PET (~18 min)**, not 50 % PET
   (~15 min) as the table heading and footnote claimed.
2. The RF figure was from the **random-search tuned Table VIII**; the LR
   and LightGBM figures were from the **un-tuned Table VII** — the rows
   silently mixed two different evaluation regimes.

**After**:

| Junior & Campelo | Random Forest | ~74% |
| Junior & Campelo | LightGBM | ~74% |
| Junior & Campelo | Logistic Regression | ~71% |

Footnote rewritten to make the PET-to-minutes interpolation explicit.

The "Three observations" paragraph was updated to "Two observations" and
now notes that Junior & Campelo sit ~3 pts lower than the @15-min
cluster because they evaluate on solo-queue ranked rather than
professional play.

### 2. Chen 2024 — unverifiable

**Before** (line 77): claimed 78.18 % LR at 20 min on Oracle's Elixir.

No peer-reviewed paper with these characteristics could be located in
any search. The closest hit
(`howarc.github.io/lol_match_analysis`) is a student class project, not a
publication. The sentence was **removed** from the Related Work
discussion.

### 3. Spaargaren citation under tidy-data schema — wrong group

**Before** (line 136): "single-row schema used by Lin (2016, Stanford
CS229; Spaargaren, 2022, Tilburg MSc thesis)".

Spaargaren's thesis actually uses a **two-row team-perspective format**
("12 datapoints per match", filtered to 2 team rows). Pairing him with
Lin under the single-row schema is incorrect. Spaargaren was **removed**
from this citation; only Lin remains.

### 4. Tsang 2025 — venue mislabelled

**Before** (line 77): cited alongside peer-reviewed work without
qualification.

Tsang is a Medium article, not peer-reviewed. The line now opens with
"an analyst writing on Medium rather than a peer-reviewed source".

### 5. Void-grub despawn timer — patch-dependent

**Before** (lines 42, 79, 410): "void grubs ... despawn at 13:45".

`13:45` was correct on Season 14 / early-2025 patches; the mid-2025
patch moved despawn to **14:45** and Rift Herald spawn from 14:00 to
**15:00**. All three mentions of `13:45` now read "13:45 on early-2025
patches; moved to 14:45 mid-season".

The 2025 Oracle's Elixir export covers patches 15.01–15.24, so both
timing regimes appear in our data.

## Game-mechanic claims — verification table

| Claim | Patch reality | Action |
|---|---|---|
| Plates pay out before 14:00 | Outer-tower plating destroyed at 14:00 — current | Kept |
| Up to 6 void grubs in pit | Confirmed | Kept |
| Void grubs spawn 5:00, despawn 13:45 | Correct early-2025; 14:45 mid-2025 | Hedged |
| Atakhan introduced 2025 | Confirmed | Kept |
| Feats of Strength introduced 2025 | Confirmed (planned removal 2026) | Kept |

## Summary of `analysis.Rmd` lines edited (this pass)

- Line 42 — void-grub timer hedge
- Lines 69–73 — Junior & Campelo numbers + footnote
- Lines 75–79 — Related Work narrative (removed Chen, downgraded Tsang
  venue, added Junior & Campelo gap explanation, added second void-grub
  hedge)
- Line 136 — Spaargaren removed from tidy-data citation
- Line 410 — void-grub timer hedge

No modelling code or numerical results were touched.
