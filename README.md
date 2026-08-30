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
| `cbrmb.adapters` | `zotus`, `taxa`, `alpha_diversity`, `filter_zotu`, `filter_tax` |
| `cbrmb.ordination` | `calc_mds`, `calc_umap` _(pending)_ |
| `cbrmb.longitudinal` | `david_recipe` and helpers _(pending)_ |
| `cbrmb.rbackend` | `unifrac`, `permanova`, `clustering`, `contingency` _(partial)_ |

The hot functions are re-exported at the top level:

```python
import cbrmb as mb

zotus = mb.filter_zotu(adata)                       # AnnData -> filtered DataFrame
res = mb.mwu_test(zotus, adata.obs["group"])        # feature-wise MWU + BH-FDR
```

## Migrating from `utils.py` / `tstools.py`

* `import utils as ut` -> `import cbrmb as mb` (the `mb.filter_zotu(adata)` calls are unchanged)
* `import tstools as ts` -> `from cbrmb import longitudinal`
* `filter_*`: keyword `return_bool` -> `return_mask`; `rel_abundance` -> `max_rel`
* `wilcoxon_test` now works (the original referenced undefined names)
* `fisher_test` on tables larger than 2x2 needs the `r` extra; 2x2 uses SciPy
