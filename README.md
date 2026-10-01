# cbrmb

Microbiome analysis helpers, extracted from a working analysis script.

Samples are rows, features (zOTUs / taxa / diversity metrics) are columns. The
statistical and filtering helpers operate on plain `pandas` DataFrames; thin
adapters pull those tables out of an `AnnData` object.

## Install

```bash
pip install -e .                 # core: numpy, pandas, scipy, statsmodels, scikit-learn
pip install -e '.[anndata]'      # AnnData adapters
pip install -e '.[umap]'         # calc_umap
pip install -e '.[r]'            # adonis / NbClust / r x c Fisher (needs R + ape, phangorn, GUniFrac, NbClust)
pip install -e '.[plotting]'     # plot_read_depth (needs matplotlib)
pip install -e '.[all,test]'
```

## Layout

| Module | Contents |
| --- | --- |
| `cbrmb.pvalues` | `fdr`, `cut_p` |
| `cbrmb.stats` | `kruskal_test`, `mwu_test`, `wilcoxon_test`, `corr_test`, `fisher_test`, `mwu_one_vs_rest`, `mwu_pairwise` |
| `cbrmb.filtering` | `filter_prevalence`, `filter_rel_abundance`, `filter_features` |
| `cbrmb.clinical` | `bernoulli_var`, `filter_bernoulli` |
| `cbrmb.adapters` | `zotus`, `taxa`, `alpha_diversity`, `filter_zotu`, `filter_tax`, `calc_gunifrac`, `best_clusters` |
| `cbrmb.unifrac` | `generalized_unifrac`, `calc_gunifrac`, `read_newick`, `root_at_midpoint` (pure Python, no R) |
| `cbrmb.ordination` | `calc_mds`, `calc_umap`, `ordination_report` (embedding + one-covariate PERMANOVA/betadisper; `.plot()` needs the `plotting` extra) |
| `cbrmb.plotting` | `plot_read_depth` (needs the `plotting` extra) |
| `cbrmb.ml` | `nested_cv`, `permutation_test`, `cross_cohort`, `KTSP`, `KTSPClassifier`, `PrevalenceThreshold`, `CLR`, `RelativeAbundance`, `RepeatedStratifiedGroupKFold` |
| `cbrmb.longitudinal` | `david_recipe` and helpers (David et al. 2014) |
| `cbrmb.rbackend.unifrac` | `calc_gunifrac` (legacy R path; `calc_gunifrac(..., backend="r")`) |
| `cbrmb.rbackend.permanova` | `subject_variation`, `screen_confounder`, `test_confounder`, `test_confounder_adjusted`, `mediation_decompose`, `bootstrap_mediation`, `screen_effect_modifiers`, `betadisper`, `remove_confounder_nan` |
| `cbrmb.rbackend.clustering` | `best_clusters` |
| `cbrmb.rbackend.contingency` | `fisher_exact_rc` (r x c fallback for `fisher_test`) |
| `cbrmb.rbackend.kernel` | `glmm_mirkat`, `cskat` (needs CRAN package `MiRKAT`) |

The hot functions are re-exported at the top level:

```python
import cbrmb as mb

zotus = mb.filter_zotu(adata)                       # AnnData -> filtered DataFrame
res = mb.mwu_test(zotus, adata.obs["group"])        # feature-wise MWU + BH-FDR
```

## Generalized UniFrac (no R)

`cbrmb.unifrac` is a pure-NumPy port of `GUniFrac::GUniFrac` plus a
`phangorn::midpoint` equivalent; it reproduces the R pipeline to machine
precision (verified against live R on real ZOTU trees). It replaces this recipe:

```r
tree <- read.tree("ZOTUs-Tree-nj.tre")
tree$tip.label <- gsub("'", "", tree$tip.label)
udist <- GUniFrac(t(norm_counts), phangorn::midpoint(tree), alpha = c(0, 0.5, 1))$unifracs[, , "d_0.5"]
```

```python
from cbrmb.unifrac import generalized_unifrac, calc_gunifrac

# samples x OTUs, same orientation as the rest of cbrmb (no transpose)
ufs   = generalized_unifrac(norm_counts, "ZOTUs-Tree-nj.tre", alphas=(0.0, 0.5, 1.0))
udist = ufs[0.5]                                    # d_0.5, a square DataFrame

udist = calc_gunifrac(norm_counts, "ZOTUs-Tree-nj.tre", alpha=0.5)   # one alpha -> one frame
udist = mb.calc_gunifrac(adata, "ZOTUs-Tree-nj.tre", alpha=0.5)      # AnnData wrapper
```

The tree is midpoint-rooted and then pruned to the OTUs in `norm_counts` (both
match R). `mb.calc_gunifrac(..., backend="r")` still runs the old rpy2 path.

## Ordination + PERMANOVA for one covariate

`ordination_report` bundles the recurring "embed a distance matrix, colour by a
group, and test that group" step: a UMAP/MDS embedding plus `test_confounder` and
`betadisper` on the *same* matrix. Pass an AnnData (uses `adata.obsp[dist_key]`,
`group`/`subject` are `obs` columns) or a bare distance matrix with `group` /
`subject` as sequences.

```python
import cbrmb as mb

for k, ax in zip(["feces", "saliva"], axs):
    sub = adata_tum[adata_tum.obs["sample_type"] == k]
    rep = mb.ordination_report(sub, dist_key="gunifrac", group="group",
                               subject=None, method="umap", min_dist=0.5)
    rep.permanova            # n, n_groups, scheme, R2, pval, betadisper_F, betadisper_pval
    rep.plot(ax=ax, colors=color_dict["group"],
             title=f"{k} (n={sub.n_obs})")   # scatter + 2D densities + legend + R2/p corner
```

`.plot()` delegates to `cbrviz.plot_embedding` (in the `plotting` extra); grab
`rep.embedding` and call `cbrviz.plot_embedding` yourself for full control.
`test=False` skips the R backend and returns just the embedding. For screening
*many* covariates at once, keep using `screen_confounder` (below).

## Repeated measures (multiple samples per subject)

See [`docs/permanova_repeated_measures.md`](docs/permanova_repeated_measures.md) for
the reasoning behind the schemes below (why free permutation is anti-conservative,
why between- and within-subject covariates need different restricted-permutation
designs). This section is the quick-reference version.

Default PERMANOVA permutes all samples freely and is **anti-conservative** when a
person contributes several samples. Pass `subject=` and `test_confounder` /
`screen_confounder` pick a restricted-permutation scheme per covariate (via the R
`permute` package):

| covariate behaviour | `level` | scheme |
| --- | --- | --- |
| constant within each subject (sex, genotype, baseline exposure) | `between` | permute whole subjects (`Plots(strata=subject, type="free")`, `Within("none")`); `subject` not in the model |
| varies within subjects (timepoint, disease activity, medication over time, sample type) | `within` / `mixed` | `subject` as first model term + permute within subject (`how(blocks=subject)`) |

```python
import cbrmb as mb

mb.subject_variation(meta, subject)                       # see how each column is classified
mb.screen_confounder(dist, meta, subject=meta["patient_id"])   # var, level, n, scheme, r2, pval, fdr
mb.test_confounder(dist, meta["sex"], subject=subj, reduce="medoid")  # or collapse to 1 sample/subject
mb.screen_effect_modifiers(dist, meta["group"], meta[["sex", "age"]], subject=subj)
mb.betadisper(dist, meta["group"], subject=subj)          # dispersion check, same scheme logic
```

For adjusted multi-covariate screening with a random subject effect there is
`cbrmb.rbackend.kernel.glmm_mirkat` (`MiRKAT::GLMMMiRKAT`; Gaussian / binomial /
Poisson outcome) and `cskat` (its Gaussian + Davies special case). Loop candidate
covariates as the outcome `y`, adjusting for the rest:

```python
mb.glmm_mirkat(dist, y=meta["disease_active"], covariates=meta[["age", "sex"]],
               subject=meta["patient_id"], model="binomial")
```

`MiRKAT` is a CRAN package (not on conda-forge):

```bash
Rscript -e 'install.packages("MiRKAT")'   # pulls CompQuadForm, GLMMadaptive, PearsonDS
```

Without `subject=`, all of these behave exactly as before.

## Mediator screening (descriptive PERMANOVA-R2 decomposition)

Does a candidate mediator (e.g. antibiotic exposure) eat into an exposure's
(e.g. disease group) community-level association? `test_confounder` gives the
unadjusted "total" R2; `test_confounder_adjusted` fits the same PERMANOVA with
the mediator entered first (sequential/Type I SS) and returns the exposure's
"direct" R2 afterwards. `mediation_decompose` runs both and reports the gap;
`bootstrap_mediation` wraps it in a cluster bootstrap for a CI on that gap.

```python
import cbrmb as mb

# total vs. direct effect of `group`, adjusting for `antibiotics`
total  = mb.test_confounder(dist, meta["group"], subject=subj)
direct = mb.test_confounder_adjusted(dist, meta["group"], meta["antibiotics"], subject=subj)

dec = mb.mediation_decompose(dist, meta["group"], meta["antibiotics"], subject=subj)
dec[["R2_total", "R2_direct", "R2_indirect", "R2_indirect_frac"]]

boot = mb.bootstrap_mediation(dist, meta["group"], meta["antibiotics"], subject=subj, n_boot=1000)
boot["summary"]   # mean/sd/CI for R2_total, R2_direct, R2_indirect
```

This is a descriptive R2 decomposition, not a formal causal-mediation
estimate (no check of the no-exposure-induced-mediator-outcome-confounder
assumption, no product-of-coefficients test) -- see the docstrings and
[`docs/permanova_repeated_measures.md`](docs/permanova_repeated_measures.md)
for the caveats and when to reach for per-taxon causal mediation instead.

## Machine learning: nested CV, cross-cohort validation, k-TSP

`cbrmb.ml` (`mb.ml`) wraps any scikit-learn estimator or pipeline in the
nested cross-validation design of Hermida, Gertz & Ruppin (Nat Commun 2022):
an outer repeated stratified k-fold (default 4 x 25 = 100 models) for honest
performance, and inside every outer training split an inner repeated CV
(default 3 x 5) that tunes the *whole* pipeline. Put every data-dependent step
(prevalence filter, transformation, feature selection, scaling) into the
pipeline so it is learned on training data only.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import cbrmb as mb
from cbrmb.ml import PrevalenceThreshold, CLR, KTSP, KTSPClassifier

enet = Pipeline([
    ("prev", PrevalenceThreshold(0.1)),          # >= 10 % of training samples
    ("clr", CLR()),
    ("sc", StandardScaler()),
    ("clf", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                               max_iter=5000)),
])
res = mb.ml.nested_cv(enet, {"clf__C": np.logspace(-3, 1, 5)}, X, y,
                      groups=patient_id,         # optional: keep a patient's samples together
                      external={"Paris": (X_paris, y_paris)},
                      metadata=meta,             # carried into predictions, not fitted
                      refit_final=True, n_jobs=8)

res.summary()             # mean / sd / interval per dataset x metric (+ locked final model)
res.scores                # one row per outer split
res.predictions           # out-of-fold + external predictions (sample, y_true, y_pred, y_score, meta)
res.best_params           # chosen hyperparameters per split
res.cv_results            # inner grid per split (e.g. AUROC vs number of features)
res.feature_stability()   # top-50 in >= 20 % of models + Holm-adjusted Wilcoxon on coefficients
res.compare(clinical_only_res)   # paired Wilcoxon over the same 100 splits
mb.ml.permutation_test(enet, grid, X, y, n_permutations=1000, observed=res)
```

`y` may be strings; the larger label (`classes_[1]`, e.g. `"PDAC"` vs
`"Control"`) is the positive class. A `DummyClassifier` with `param_grid=None`
gives the chance baseline over the same splits.

**Cross-cohort validation** trains in each cohort (with nested CV inside it)
and validates in the others; `scheme="loco"` trains on all but one cohort
(stratified by label x cohort) and validates on the held-out one:

```python
cc = mb.ml.cross_cohort(pipe, grid, X, y, cohort=obs["cohort"],
                        scheme=["pairwise", "loco"], groups=patient_id, n_jobs=8)
cc.matrix("roc_auc")                 # train x test; diagonal = internal nested CV
cc.matrix("roc_auc", kind="final")   # locked model refit on the whole training cohort
cc.summary("roc_auc")                # mean / sd / interval per train x test
cc.results["TUM"]                    # the full NestedCVResult behind each row
```

**k top scoring pairs** (`KTSP`, `KTSPClassifier`) is a pure-NumPy port of
Bioconductor `switchBox` (`SWAP.Train.KTSP`). Pairs, scores, tie votes and
pair indicators are byte-identical to R (tested against switchBox 1.44,
including `handleTies` and the Wilcoxon pre-filter), and it is 10-25x
faster. Because only the within-sample order of two taxa matters, k-TSP rules
do not depend on sequencing depth or normalisation, which helps across cohorts.

```python
# pairs as features for any classifier, k tuned like any hyperparameter
ktsp_lgr = Pipeline([("prev", PrevalenceThreshold(0.1)),
                     ("ktsp", KTSP(handle_ties=True)),        # handle_ties for zero-inflated counts
                     ("clf", LogisticRegression())])
grid = {"ktsp__k": [5, 10, 25], "clf__C": [0.01, 0.1, 1]}

# or the classic majority-vote classifier; a list for k = switchBox krange
vote = Pipeline([("prev", PrevalenceThreshold(0.1)),
                 ("ktsp", KTSPClassifier(k=list(range(3, 26, 2)), handle_ties=True))])

fit = vote.fit(X, y)
fit[-1].get_pairs(fit[:-1].get_feature_names_out())   # a, b, comparison, score, tie_vote
```

Differences from switchBox, all on purpose: when there are fewer disjoint pairs
than `k`, switchBox fills up with self-pairs or `NA` and we stop instead.
Choosing `k` from a range uses the same comparison as prediction (switchBox
quietly uses `a >= b` there). Feature names show the comparison that is
actually evaluated.

## Migrating from `utils.py` / `tstools.py`

* `import utils as ut` -> `import cbrmb as mb`. Unchanged calls: `filter_zotu`,
  `filter_tax`, `mwu_test`, `kruskal_test`, `fisher_test`, `mwu_pairwise`,
  `screen_confounder`, `test_confounder`, `best_clusters`, `cut_p`, `fdr`.
* `import tstools as ts` -> `import cbrmb.longitudinal as ts`
  (`ts.david_recipe(df)` is unchanged; helper `topN_otu_indices` renamed to
  `top_otu_indices`, kwarg `topN` -> `top_frac`).
* `filter_*`: keyword `return_bool` -> `return_mask`; `rel_abundance` -> `max_rel`.
* `calc_umap` / `calc_mds`: `seed` -> `random_state`; new `index=` argument fills
  the row labels, so `calc_umap(adata.obsp["gunifrac"], index=adata.obs_names)`
  replaces the `.set_index(adata.obs_names)` follow-up.
* `wilcoxon_test` now works (the original referenced undefined names).
* `fisher_test` on tables larger than 2x2 needs the `r` extra; 2x2 uses SciPy.
* `fdr` column now sits directly after `pval` in every `*_test` result.
* Dropped (broken / unused): `test_confounder_maaslin`, `rpy2fisher`, `lme4`.
