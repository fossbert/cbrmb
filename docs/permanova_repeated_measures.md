# Repeated-Measures PERMANOVA for 16S Beta-Diversity

This note explains the statistics behind `cbrmb.rbackend.permanova`
(`test_confounder`, `screen_confounder`, `subject_variation`, `betadisper`) and
when to reach for which option. The README's "Repeated measures" section is
the quick-reference table; this is the "why" behind it.

## 1. The problem: pseudoreplication

PERMANOVA (`vegan::adonis2` / `GUniFrac::adonis3`) tests a covariate against a
distance matrix by comparing the observed pseudo-F to a null distribution
built from permuting sample labels. That null distribution is only valid if
every sample is *exchangeable* with every other sample under the null.

When several samples come from the same patient (repeated visits, several body
sites, technical replicates), that assumption breaks: samples from one patient
are more similar to each other than to samples from a different patient,
independent of any covariate. A free permutation ignores this and treats each
sample as an independent unit of evidence, which inflates the effective sample
size and produces anti-conservative (too small) p-values. `tests/test_permanova_rm.py`
demonstrates this directly (`test_free_permutation_is_anticonservative_vs_restricted`):
the same effect, tested free vs. subject-aware, gives the same R²/F but a much
larger, correct p-value once the repeated-measures structure is respected.

The fix is not a different test statistic — R² and pseudo-F stay the same —
but a *restricted permutation scheme* that only ever compares samples in ways
that are actually exchangeable under the null.

## 2. Step 0: classify the covariate relative to the subject

The correct restriction depends on whether the covariate you're testing is
constant within each patient or varies across their visits:

| covariate behaviour | example | `subject_variation` level |
| --- | --- | --- |
| one fixed value per patient | sex, genotype, baseline diagnosis | `between` |
| changes across a patient's own samples | visit/timepoint, disease activity, medication started mid-study, sample type | `within` (or `mixed` if only some patients vary) |
| identical for everyone | a batch that never changed | `constant` — untestable |
| unique per sample | a sample barcode | `id_like` — untestable |

`subject_variation(meta, subject)` runs this classification for every column
of a metadata table (see `cbrmb/rbackend/permanova.py:_classify`), and
`test_confounder(..., scheme="auto")` uses it to pick the scheme below without
you having to decide by hand.

This distinction matters because the two cases need *different* restricted
permutations, not the same one applied loosely.

## 3. Between-subject covariates: permute whole patients

If the covariate is fixed per patient, patient identity and the covariate are
confounded — you cannot shuffle a between-subject covariate *within* one
patient's samples (there is only one value to shuffle). The valid null instead
permutes **whole patients** as indivisible units: reassign each patient's
entire block of samples to a different covariate label, but never break up or
reshuffle samples within a patient.

In `permute` terms this is:

```r
permute::how(
  plots  = Plots(strata = subject, type = "free"),  # patients are the permutable units
  within = Within(type = "none")                    # never reshuffle within a patient
)
```

implemented as `_how_between()`. The model formula stays `y ~ x` (subject is
*not* added as a term — it would be collinear with a between-subject `x`).

## 4. Within-subject covariates: block by subject and adjust for it

If the covariate varies across a patient's own visits, the natural null
*keeps each patient's own set of samples together* and only asks whether the
within-patient ordering/labelling of the covariate explains more of that
patient's own variance than chance. This is a same-patient-only restriction:

```r
permute::how(blocks = subject)   # only ever permute within one patient's samples
```

implemented as `_how_blocks()`. Here `subject` **is** added to the model
(`y ~ subject + x`, sequential/Type I sums of squares), so `x` is tested on
the residual variance *after* accounting for between-patient differences —
i.e. purely on the within-patient signal, which is what the blocked
permutation is actually the null distribution for.

## 5. Practical recipe

```python
import cbrmb as mb

# 1. see how each covariate behaves relative to patient
mb.subject_variation(meta, subject=meta["patient_id"])

# 2. test one covariate; scheme is picked automatically
res = mb.test_confounder(dist, meta["diagnosis"], subject=meta["patient_id"])
res["R2"], res["Pr(>F)"], res.attrs["scheme"]   # scheme tells you what was actually run

# 3. screen many covariates at once, with FDR correction
tab = mb.screen_confounder(dist, meta[["diagnosis", "sex", "visit", "crp"]],
                            subject=meta["patient_id"])
```

Force a scheme explicitly with `scheme="between"` / `"within"` if the
automatic classification is ambiguous (e.g. a `mixed` covariate you want to
treat conservatively as `within`).

## 6. When restricted permutation isn't the right tool

* **Very unbalanced designs** (patients with wildly different numbers of
  visits, or only 2–3 patients per arm) make the permutation universe small
  and the restricted test underpowered. `reduce="medoid"` or `reduce="first"`
  collapses each patient to one representative sample (medoid = the sample
  closest to that patient's own centroid) and runs an ordinary PERMANOVA on
  patient-level units — simpler, fully valid, but discards the longitudinal
  resolution. With exactly two samples per patient (pre/post) both are equally
  close to the patient's centroid; the tie goes to the sample closest to *all*
  samples, so the result never depends on how the input rows are sorted.
  `reduce="first"` keeps the first row per patient — sort by visit beforehand
  if "first" should mean baseline.
* **Adjusting for several covariates at once with a genuine random subject
  effect** (rather than one covariate at a time) is a different model class —
  see `cbrmb.rbackend.kernel.glmm_mirkat` (`MiRKAT::GLMMMiRKAT`), which fits a
  GLMM-kernel test with `subject` as a random effect and arbitrary fixed-effect
  adjustment.
* **Interaction / effect-modification** (does covariate A's effect depend on
  B?) is `screen_effect_modifiers`, which fits `subject + A * B` and picks the
  permutation scheme from whichever of A/B is "more within-subject".

## 7. Don't skip the dispersion check

A significant PERMANOVA can reflect a difference in group *location*
(centroids) or in group *spread* (dispersion) — `adonis`/`adonis2` alone
cannot tell these apart. Run `betadisper(dist, group, subject=subject)`
alongside any PERMANOVA result you plan to report; it applies the same
subject-restricted permutation logic to `vegan::betadisper`/`permutest`, so a
significant PERMANOVA together with a non-significant `betadisper` supports a
genuine location effect.

## 8. Mediator screening: does a candidate mediator eat into an exposure's R2?

A recurring question once `screen_confounder` flags several candidates: one
of them (say, antibiotic exposure) looks like it might sit *causally between*
the exposure of interest (say, disease group) and the community, rather than
being an independent confounder -- exposure causes the mediator, which in
turn shifts the community. `test_confounder`/`screen_confounder` alone can't
tell you this: they test one covariate at a time, unconditionally.

`test_confounder_adjusted(dist, target, adjust_for, ...)` fits the same
PERMANOVA but with `adjust_for` entered first in a sequential (Type I) model
-- `y ~ [subj +] adjust_for + target` -- and returns `target`'s R2 *after*
`adjust_for` has already claimed its share. With `subject=`, the scheme
(whole-subject vs. within-subject-blocked permutation) is decided from
`target`'s own level exactly as in `test_confounder` -- `adjust_for` never
drives that choice, it only ever enters as a control term. Comparing that to
the plain, unadjusted `test_confounder(dist, target, ...)` gives a rough
decomposition:

```
R2_total    = test_confounder(dist, exposure, ...)                    # exposure alone
R2_direct   = test_confounder_adjusted(dist, exposure, mediator, ...) # exposure after mediator
R2_indirect = R2_total - R2_direct                                    # ~ shared with mediator
```

`mediation_decompose` runs both calls on the same joint-complete-case subset
(so the two R2 values are comparable) and returns all of `R2_total`,
`R2_direct`, `R2_indirect` and `R2_indirect_frac` in one call.
`bootstrap_mediation` wraps this in a cluster bootstrap (resampling subjects
with replacement when `subject=` is given, samples otherwise) to put a
percentile CI on `R2_indirect`. Each bootstrap replicate runs with `n_perm=2`
-- the minimum `GUniFrac::adonis3` needs to build a well-formed ANOVA table
for the two-covariate `m + x` formula (`n_perm=1` makes it error) -- since the
R2/pseudo-F point estimate does not otherwise depend on the permutation count;
only its own p-value would, and that p-value is discarded, so this still keeps
1000+ replicates fast. Only `bootstrap_mediation(...)["observed"]` (fit with
the real `n_perm`) has a meaningful p-value. A subject drawn more than once in a
replicate is relabelled per draw (`f"{subject}__{k}"`) so the `within`/
`between` restricted-permutation machinery treats each drawn copy as its own
unit, as a cluster bootstrap requires.

**What this is not.** It is a descriptive variance-share comparison, useful
as a fast, assumption-light screen before investing further -- not a formal
causal mediation estimate. It doesn't check whether there's a confounder of
the mediator-outcome relationship that is itself affected by the exposure
(a common real case: exposure causes both the mediator and a second variable
that also shifts the community, e.g. disease severity driving both medication
use and a physiological change that independently shifts the community).
R2 shares from sequential SS are also not strictly additive/causal
quantities in the way a coefficient decomposition (a x b) would be, and a
`between`-level `adjust_for` inside a `within`-blocked model is aliased by
the `subj` term (see `test_confounder_adjusted`'s docstring). Treat a
nonzero `R2_indirect` here as "worth following up", not as a mediation
finding to report -- for that, move to per-taxon or per-metric causal
mediation (R: `mediation`, `CMAverse`, `regmedint`) with the usual
sensitivity analysis for unmeasured mediator-outcome confounding.

**Where this sits in the literature.** None of the pieces below is new; the
decomposition just combines them for distance-based community data.

* *R2 split.* `R2_total - R2_direct` is the distance-matrix analogue of the
  "difference in coefficients" estimator (c - c') of the indirect effect, which
  MacKinnon et al. (2002) found to perform as well as product-of-coefficients
  estimators. In variation-partitioning terms (Peres-Neto et al. 2006) it is the
  fraction of the exposure's R2 *shared* with the mediator. Those authors also
  show that raw (unadjusted) R2 fractions are biased, which is why small
  `R2_indirect` values near zero (including slightly negative ones) should be
  read against the bootstrap CI rather than at face value.
* *Bootstrap CI.* The indirect effect is a difference/product of estimates and
  is not normally distributed, so resampling CIs are preferred over a
  normal-theory standard error (Shrout & Bolger 2002; MacKinnon et al. 2004).
  `bootstrap_mediation` uses the plain percentile interval. Resampling whole
  subjects is the cluster bootstrap, which Field & Welsh (2007) show to be
  consistent for clustered data under a random-effects structure.
* *Distance-based mediation tests.* If you need a formal global test rather
  than a descriptive share, `MedTest` (Zhang et al. 2018) is the closest
  distance-matrix counterpart. Per-taxon alternatives with FDR control are
  `LDM-med` (Yue & Hu 2022) and compositional mediation (Sohn & Li 2019).
* *Causal interpretation.* The assumptions behind reading `R2_indirect` as a
  mediated effect (no unmeasured exposure-mediator, mediator-outcome or
  exposure-outcome confounding; Imai et al. 2010) and the exposure-induced
  mediator-outcome confounder problem described above (VanderWeele et al. 2014)
  are exactly what the R packages `mediation` (Tingley et al. 2014) and
  `CMAverse` (Shi et al. 2021) address with sensitivity analyses.
* *Why medication is the usual candidate.* Drug exposure is a well-documented
  source of spurious case/control microbiome differences (Forslund et al. 2015;
  Vujkovic-Cvijin et al. 2020), and antibiotic effects are time-dependent and
  often incompletely reversed (Dethlefsen & Relman 2011). The exposure window
  used to code a mediator therefore matters, and the coding should be defined
  the same way in all groups.

## References

* Anderson, M.J. & ter Braak, C.J.F. (2003). *Permutation tests for
  multi-factorial analysis of variance.* J. Stat. Comput. Simul.
* McArdle, B.H. & Anderson, M.J. (2001). *Fitting multivariate models to
  community data: a comment on distance-based redundancy analysis.* Ecology.
* Anderson, M.J. (2001). *A new method for non-parametric multivariate
  analysis of variance.* Austral Ecology (the original PERMANOVA paper).
* `permute` package vignette (Simpson, G.L.) — `Plots`/`Within`/`blocks`
  restricted-permutation design used here.
* Zhan, X. et al. (2018). *A fast small-sample kernel independence test for
  microbiome community-level association analysis* / MiRKAT-family GLMM
  extension used by `glmm_mirkat`.

### Mediation decomposition (section 8)

*Distance-based regression and R2 partitioning*

* Zapala, M.A. & Schork, N.J. (2006). *Multivariate regression analysis of
  distance matrices for testing associations between gene expression patterns
  and related variables.* PNAS 103(51), 19430-19435.
  doi:10.1073/pnas.0609333103
* Peres-Neto, P.R., Legendre, P., Dray, S. & Borcard, D. (2006). *Variation
  partitioning of species data matrices: estimation and comparison of
  fractions.* Ecology 87(10), 2614-2625.

*Estimating and testing an indirect effect*

* MacKinnon, D.P., Lockwood, C.M., Hoffman, J.M., West, S.G. & Sheets, V.
  (2002). *A comparison of methods to test mediation and other intervening
  variable effects.* Psychological Methods 7(1), 83-104.
* Shrout, P.E. & Bolger, N. (2002). *Mediation in experimental and
  nonexperimental studies: new procedures and recommendations.* Psychological
  Methods 7(4), 422-445.
* MacKinnon, D.P., Lockwood, C.M. & Williams, J. (2004). *Confidence limits for
  the indirect effect: distribution of the product and resampling methods.*
  Multivariate Behavioral Research 39(1), 99-128.
  doi:10.1207/s15327906mbr3901_4
* Field, C.A. & Welsh, A.H. (2007). *Bootstrapping clustered data.* J. R.
  Stat. Soc. B 69(3), 369-390. doi:10.1111/j.1467-9868.2007.00593.x

*Mediation methods for microbiome data*

* Zhang, J., Wei, Z. & Chen, J. (2018). *A distance-based approach for testing
  the mediation effect of the human microbiome.* Bioinformatics 34(11),
  1875-1883.
* Sohn, M.B. & Li, H. (2019). *Compositional mediation analysis for microbiome
  studies.* Annals of Applied Statistics 13(1), 661-681.
  doi:10.1214/18-AOAS1210
* Yue, Y. & Hu, Y.-J. (2022). *A new approach to testing mediation of the
  microbiome at both the community and individual taxon levels.*
  Bioinformatics 38(12), 3173-3180. doi:10.1093/bioinformatics/btac310

*Causal mediation (what the descriptive R2 split is not)*

* Imai, K., Keele, L. & Tingley, D. (2010). *A general approach to causal
  mediation analysis.* Psychological Methods 15(4), 309-334.
  doi:10.1037/a0020761
* VanderWeele, T.J., Vansteelandt, S. & Robins, J.M. (2014). *Effect
  decomposition in the presence of an exposure-induced mediator-outcome
  confounder.* Epidemiology.
* Tingley, D., Yamamoto, T., Hirose, K., Keele, L. & Imai, K. (2014).
  *mediation: R package for causal mediation analysis.* Journal of Statistical
  Software 59(5), 1-38. doi:10.18637/jss.v059.i05
* Shi, B., Choirat, C., Coull, B.A., VanderWeele, T.J. & Valeri, L. (2021).
  *CMAverse: a suite of functions for reproducible causal mediation analyses.*
  Epidemiology 32(5), e20-e22. doi:10.1097/EDE.0000000000001378

*Medication as confounder/mediator in microbiome case-control studies*

* Forslund, K. et al. (2015). *Disentangling type 2 diabetes and metformin
  treatment signatures in the human gut microbiota.* Nature 528, 262-266.
  doi:10.1038/nature15766
* Vujkovic-Cvijin, I. et al. (2020). *Host variables confound gut microbiota
  studies of human disease.* Nature 587, 448-454.
  doi:10.1038/s41586-020-2881-9
* Dethlefsen, L. & Relman, D.A. (2011). *Incomplete recovery and individualized
  responses of the human distal gut microbiota to repeated antibiotic
  perturbation.* PNAS 108(Suppl. 1), 4554-4561.
  doi:10.1073/pnas.1000087107
