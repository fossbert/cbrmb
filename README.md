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
pip install -e '.[r]'            # UniFrac / adonis / NbClust / r x c Fisher (needs R + ape, phangorn, GUniFrac, NbClust)
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
| `cbrmb.ordination` | `calc_mds`, `calc_umap` |
| `cbrmb.longitudinal` | `david_recipe` and helpers (David et al. 2014) |
| `cbrmb.rbackend.unifrac` | `calc_gunifrac` |
| `cbrmb.rbackend.permanova` | `subject_variation`, `screen_confounder`, `test_confounder`, `screen_effect_modifiers`, `betadisper`, `remove_confounder_nan` |
| `cbrmb.rbackend.clustering` | `best_clusters` |
| `cbrmb.rbackend.contingency` | `fisher_exact_rc` (r x c fallback for `fisher_test`) |
| `cbrmb.rbackend.kernel` | `cskat`, `glmm_mirkat` (needs R package `MiRKAT`) |

The hot functions are re-exported at the top level:

```python
import cbrmb as mb

zotus = mb.filter_zotu(adata)                       # AnnData -> filtered DataFrame
res = mb.mwu_test(zotus, adata.obs["group"])        # feature-wise MWU + BH-FDR
```

## Repeated measures (multiple samples per subject)

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
`cbrmb.rbackend.kernel.cskat` / `glmm_mirkat` (CSKAT / GLMM-MiRKAT). These need
the R package `MiRKAT`:

```bash
mamba install -n microbiome -c conda-forge r-mirkat   # or, in R: install.packages("MiRKAT")
```

Without `subject=`, all of these behave exactly as before.

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
