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
| `cbrmb.rbackend.permanova` | `screen_confounder`, `test_confounder`, `remove_confounder_nan` |
| `cbrmb.rbackend.clustering` | `best_clusters` |
| `cbrmb.rbackend.contingency` | `fisher_exact_rc` (r x c fallback for `fisher_test`) |

The hot functions are re-exported at the top level:

```python
import cbrmb as mb

zotus = mb.filter_zotu(adata)                       # AnnData -> filtered DataFrame
res = mb.mwu_test(zotus, adata.obs["group"])        # feature-wise MWU + BH-FDR
```

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
