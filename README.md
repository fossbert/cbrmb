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
pip install -e '.[r]'            # adonis / NbClust / r x c Fisher (needs R + ape, phangorn, GUniFrac, NbClust; lme4, lmerTest, nlme for mixed models)
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
| `cbrmb.paired` | `find_pairs` -> `Pairs` (`.delta`, `.wide`, `.within_distance`, `.distance_matrix`, `.shift_vectors`, `.shift_distance`, `.delta_distance`, `.subset`, `.split`, `.long`), `align_samples`, `example_data` |
| `cbrmb.distance` | `as_distance_frame`, `subset_distance`, `distance_matrix`, `pcoa`, `mantel_test`, `mantel_screen`, `within_group_distances` |
| `cbrmb.rbackend.unifrac` | `calc_gunifrac` (legacy R path; `calc_gunifrac(..., backend="r")`) |
| `cbrmb.rbackend.permanova` | `subject_variation`, `screen_confounder`, `test_confounder`, `test_confounder_adjusted`, `mediation_decompose`, `bootstrap_mediation`, `screen_effect_modifiers`, `betadisper`, `remove_confounder_nan` |
| `cbrmb.rbackend.clustering` | `best_clusters` |
| `cbrmb.rbackend.contingency` | `fisher_exact_rc` (r x c fallback for `fisher_test`) |
| `cbrmb.rbackend.kernel` | `glmm_mirkat`, `cskat` (needs CRAN package `MiRKAT`) |
| `cbrmb.rbackend.mixed` | `alpha_mixed`, `alpha_mixed_screen`, `AlphaMixedFit` (alpha-diversity mixed models; needs `lme4`, `lmerTest`, `nlme`) |

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

## Alpha diversity in longitudinal data (mixed models)

`cbrmb.rbackend.mixed` fits one mixed model per alpha-diversity measure, with a
random effect per subject. The family follows from the measure:

| measure | `family` | model | `exp(beta)` |
| --- | --- | --- | --- |
| Shannon effective number (`exp(H)`) | `"lognormal"` | `lmerTest::lmer(log(y) ~ ...)`, Satterthwaite df | ratio of geometric means |
| (rarefied / observed) richness | `"negbin"` | `lme4::glmer.nb`, log link | rate ratio |
| underdispersed counts | `"poisson"` | `lme4::glmer` | rate ratio |

`family="auto"` picks `negbin` for non-negative integers, otherwise `lognormal`.

```python
import cbrmb as mb

df = adata.obs.join(adata.obsm["alpha_diversity"])      # one row per sample

fit = mb.alpha_mixed(df, "Shannon.Effective", "group * time + age", subject="MT-ID",
                     random="slope", time="time")        # (1 + time | MT-ID)
fit.coef      # estimate, se, df, stat, pval, ci_low/high, ratio, ratio_low/high
fit.terms     # marginal term tests (here only group:time)
fit.random    # random-effect / residual SDs
fit.info      # n_obs, n_subjects, aic, singular, theta, icc, formula, ...
fit.predict(pd.DataFrame({"group": ["ctrl", "IBD"] * 2, "time": [0, 0, 1, 1]}))

rich = mb.alpha_mixed(df, "Richness", "group * time", "MT-ID", depth="Nreads")

# several measures at once, BH-FDR per term across measures
coef, terms, fits = mb.alpha_mixed_screen(
    df, {"Shannon.Effective": "lognormal", "Normalized.Richness": "negbin"},
    "group * time", "MT-ID", depth="Nreads")
```

* **Random effects:** `random="intercept"` (default), `"slope"` (needs `time=`),
  or any lme4 term string such as `"(1 | MT-ID) + (1 | site)"`.
* **Serial correlation:** `correlation="car1"` (lognormal / gaussian) switches to
  `nlme::lme` with a continuous-time AR(1), which also handles irregular visit spacing.
* **Sequencing depth:** `depth=` adds centered `log(depth)` as a covariate;
  `depth_as="offset"` (count families) models diversity per read instead.
  Rarefied measures need neither.
* **Term tests** respect marginality: Satterthwaite F for `lmer`, otherwise
  likelihood-ratio tests from ML refits without the term.
* R convergence / singular-fit messages come back as a Python `RuntimeWarning`
  and in `fit.warnings`; `fit.r_fit` is the raw R model. If a `negbin` fit
  reports a huge `theta`, the counts are not overdispersed -- use `"poisson"`.

`glmmTMB` would add beta / Gamma families and more correlation structures, but
building it from source needs `gfortran` and BLAS/LAPACK headers; it is not
used here.

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

## Paired samples: deltas, within-pair distances, Mantel

The recurring chore: the metadata says which patient and visit each sample
belongs to, and you need *per patient* the change between visits -- of alpha
diversity, of every zOTU, of a covariate -- or the UniFrac distance between the
two samples, or a patient x patient matrix of microbiome shifts for a Mantel
test. `cbrmb.paired` does the matching once; everything after that is one call.

**The model:** `find_pairs` takes the sample-level metadata (one row per sample,
indexed by sample ID) and returns a `Pairs` object. Its methods take any
*sample-indexed* data -- DataFrame, Series, AnnData, distance matrix -- and return
*patient-indexed* results that line up with each other by index.

### Try it on the bundled toy data

```python
>>> import cbrmb as mb
>>> meta, alpha, counts = mb.paired.example_data()   # 3 patients, stool + saliva, visits 1/2
>>> pairs = mb.find_pairs(meta, unit="case_id", visit="Visit", visits=[1, 2], by="sample_type")
>>> pairs
Pairs(4 units by ['case_id', 'sample_type']; samples per Visit -> 1: 4, 2: 4)
  feces: 2
  saliva: 2
>>> pairs.table                      # p3 has no visit-2 stool sample -> no stool pair
Visit                 1    2
case_id sample_type
p1      feces        s1   s2
        saliva       s6   s7
p2      feces        s3   s4
p3      saliva       s9  s10
>>> stool = pairs.subset(sample_type="feces")      # index is now just case_id
>>> stool.delta(alpha)                             # visit 2 - visit 1
         shannon  richness
case_id
p1          -0.5     -25.0
p2           0.4      11.0
>>> stool.delta(counts, transform="rel").round(2)  # change in relative abundance
         otuA  otuB
case_id
p1        0.2  -0.2
p2        0.0   0.0
>>> stool.wide(alpha, "shannon")                   # side by side, for paired plots
Visit      1    2
case_id
p1       3.0  2.5
p2       4.0  4.4
>>> D = mb.distance_matrix(alpha, standardize=True)  # any sample x sample matrix
>>> stool.within_distance(D).round(2)
case_id
p1    0.88
p2    0.57
Name: distance_1_2, dtype: float64

```

### Real data: microbiome shift vs. QoL change and antibiotics (PDAC)

```python
import cbrmb as mb

# 1. pairs: per patient and sample type, visits 1 and 2;
#    a re-sequenced sample in the same slot -> keep the deeper one
pairs = mb.find_pairs(meta, unit="case_id", visit="Visit", visits=[1, 2],
                      by="sample_type", prefer="nreads")
stool = pairs.subset(sample_type="feces")

# 2. per-patient changes -- straight from the AnnData
d_alpha = stool.delta(adata, ["Shannon.Effective", "Richness"], obsm="alpha_diversity")
d_clr   = stool.delta(adata, layer="raw_counts", transform="clr")   # CLR over all zOTUs
uf_v1v2 = stool.within_distance(adata, "gunifrac")                  # UniFrac V1 vs V2

mb.corr_test(qol_delta.loc[d_alpha.index], d_alpha["Shannon.Effective"])
mb.corr_test(qol_delta.loc[uf_v1v2.index], uf_v1v2)

# 3. Mantel: does the microbiome move in the same direction in patients whose QoL changed alike?
shift = stool.shift_distance(adata, "gunifrac")       # PCoA -> V2-V1 vectors -> patient x patient
mb.mantel_test(shift, mb.distance_matrix(qol_delta, standardize=True), random_state=42)
mb.mantel_screen(shift, qol_delta, random_state=42)   # one test per domain, + FDR

# 4. covariates that are keyed by mt_id + visit instead of sample ID
cov = mb.align_samples(covars.assign(sample_type="feces"), meta,
                       on={"mt_id_stool": "mt_id", "sample_type": "sample_type", "visit": "Visit"})
# align_samples: 57/58 rows matched to a sample; 1 without a sample: MTGJF3471/feces/3
abx = stool.delta_distance(cov, "antibiotics", metric="cityblock")  # |delta_i - delta_j|
mb.mantel_test(shift, abx)                            # aligns on the shared patients itself

# 5. stool shift vs. saliva shift, and the same for every sample type in a loop
shifts = {st: p.shift_distance(adata, "gunifrac") for st, p in pairs.split().items()}
mb.mantel_test(shifts["feces"], shifts["saliva"])

# 6. repeated-measures PERMANOVA on just the paired samples
d = stool.distance_matrix(adata, "gunifrac")
mb.screen_confounder(d, cov.loc[d.index, ["visit", "antibiotics", "ppi"]], subject=stool.long()["case_id"])
```

### What `Pairs` gives you

| method | result | typical use |
| --- | --- | --- |
| `delta(data, cols, transform=)` | patient x feature, `late - early` | alpha / zOTU / covariate change; `transform="clr"`, `"log2"`, `"rel"` |
| `wide(data, col)` | patient x visit | `PairedStripBox` |
| `within_distance(dist, key)` | patient -> d(early, late) | how far did the microbiome move |
| `shift_vectors(dist, key)` / `shift_distance(...)` | patient x PCo / patient x patient | direction of change; Mantel input |
| `delta_distance(data, cols)` | patient x patient distance of deltas | Mantel input (QoL, binary covariates) |
| `distance_matrix(dist, key)` + `long()` | sample x sample sub-matrix + sample -> patient/visit | PERMANOVA with `subject=` |
| `subset(sample_type=...)`, `split()` | `Pairs` per sample type | one loop instead of copy-pasted cells |
| `summary()`, `samples`, `complete()` | counts, sample IDs, the pair table | QC, `adata[pairs.samples]` |

* **Duplicates** (re-runs, dilutions in the same patient-visit slot) warn and
  list the clashes; `prefer="nreads"` keeps the deepest sample, `duplicates="error"` raises.
* **More than two visits:** `visits=[1, 2, 3], min_visits=2`; every method takes
  `early=` / `late=` (default first and last visit).
* **Missing data** (a paired sample absent from the abundance table or the
  distance matrix) drops that patient with a warning naming the samples -- never silently.
* **Key matching** in `align_samples` treats `1`, `1.0` and `"1"` as equal and
  reports unmatched rows; ambiguous keys raise.

`cbrmb.distance` holds the label-aware building blocks: `pcoa`, `mantel_test`
(returns `MantelResult(r, p, n)`, unpacks like a tuple, vectorized
permutations), `mantel_screen`, `distance_matrix` (any `pdist` metric,
`standardize=`), `subset_distance`, `within_group_distances` (all within-patient
distances, for dense time series).

Every docstring in `cbrmb.paired` / `cbrmb.distance` carries a runnable example
(checked as doctests in the test suite); `help(mb.Pairs)` is a good start.

### Before / after

The stool part of the PDAC Mantel analysis used to be ~25 lines across seven
cells: composite string keys (`mt_id + "_feces_" + visit.astype(int).astype(str)`)
for `order_by_overlap`, `find_matched_visits(...).reset_index()...set_index()`,
an `np.where(adata.obs_names == s)` loop for the sub-matrix, PCoA, a per-patient
`iterrows` loop for the shift vectors, a distance matrix, `index.intersection`,
and Mantel -- then everything copied for saliva. Now:

```python
cov = mb.align_samples(covars_stool.assign(sample_type="feces"), meta,
                       on={"mt_id_stool": "mt_id", "sample_type": "sample_type", "visit": "Visit"})
stool = mb.find_pairs(meta.loc[cov.index], "case_id", "Visit", [1, 2])
mb.mantel_test(stool.shift_distance(adata, "gunifrac"),
               stool.delta_distance(cov, "antibiotics", metric="cityblock"), random_state=42)
```

Verified on the PDAC data against the old `utils.py` recipe: shift distances
agree to 1e-15; within-pair UniFrac, alpha and antibiotic deltas are identical;
the Mantel r is identical (p differs only by the random permutation stream) and
the test runs ~70x faster.

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
* Paired-sample helpers: `find_matched_visits` + `compute_visit_deltas` ->
  `find_pairs(...).delta(...)`; `order_by_overlap` / `align_on_overlap` with
  composite string keys -> `align_samples`; `classical_pcoa` -> `pcoa`;
  `pairwise_distance_matrix` -> `distance_matrix` (any `pdist` metric);
  `binary_delta_distance` -> `Pairs.delta_distance(..., metric="cityblock")`;
  `get_group_distances` -> `within_group_distances`; `mantel_test(seed=)` ->
  `mantel_test(random_state=)`.
