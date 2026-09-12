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
  resolution.
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
