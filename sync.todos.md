TODOS:
- Align H2 -> transform to something new / verify we can compare models only on 1 hypothesis in S1
- Check the diagram for Objective / side corelation in EDA
- Check hyperparams -> docs/HYPERPARAMS.md
- Walk through docs/LOGIC_CHECK_2026-04-26_2249.md (Sensitivity fix done; remaining: feature-count 46/21, EDA Summary pt5 numbers, set.seed for CV/RF, RF uplift claim attribution, plate_diff r=0.409, Shiny red-flag inversion, KNN unique/odd order, duplicate side prop test, Conclusions TODOs, line 341 TODO)
- Per-lane k/d/a enrichment (proposed S1 part 2): tested -> not worth adding. H3 = H2 + 20 role kills/deaths/assists diffs gains +0.0043 AUC over H1 (within noise, SE ≈ ±0.01); every new feature ranks 46-69/70. See docs/ROLE_ENRICHMENT_EXPERIMENT_2026-04-26_2338.md
- ...