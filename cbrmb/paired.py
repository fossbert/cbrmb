"""Paired / repeated samples: find them once, then derive everything from them.

The usual chore: a sample-level metadata table (one row per sample, indexed by
sample ID -- ``adata.obs`` or the clinical sheet) says which patient and visit
each sample belongs to, and you need, per patient, the change from visit 1 to
visit 2 -- of alpha diversity, of every zOTU, of a covariate -- or the
microbiome distance between the two samples, or a patient x patient matrix of
microbiome *shifts* for a Mantel test.

:func:`find_pairs` does the matching once and returns a :class:`Pairs` object:
one row per unit (patient, or patient x sample type), one column per visit,
holding the sample ID found there. Its methods take any *sample-indexed* data
(DataFrame, Series, AnnData, distance matrix) and return *unit-indexed*
results, which line up with each other -- and with other per-patient tables
such as QoL deltas -- by index. No positional bookkeeping, no ``np.where``.

Walkthrough on the bundled toy data (3 patients, stool + saliva, visits 1/2;
p3 has no visit-2 stool sample, so it only appears in the saliva pairs):

>>> import cbrmb as mb
>>> meta, alpha, counts = mb.paired.example_data()
>>> pairs = mb.find_pairs(meta, unit="case_id", visit="Visit", visits=[1, 2],
...                       by="sample_type")
>>> pairs
Pairs(4 units by ['case_id', 'sample_type']; samples per Visit -> 1: 4, 2: 4)
  feces: 2
  saliva: 2
>>> pairs.table
Visit                 1    2
case_id sample_type
p1      feces        s1   s2
        saliva       s6   s7
p2      feces        s3   s4
p3      saliva       s9  s10

Deltas (visit 2 - visit 1) from any sample-indexed table:

>>> pairs.delta(alpha)
                     shannon  richness
case_id sample_type
p1      feces           -0.5     -25.0
        saliva           0.6      15.0
p2      feces            0.4      11.0
p3      saliva           0.5      12.0

Drop the sample-type level to get one row per patient:

>>> stool = pairs.subset(sample_type="feces")
>>> stool.delta(alpha, "shannon")
case_id
p1   -0.5
p2    0.4
Name: shannon, dtype: float64

Distances: any labelled sample x sample matrix (here Euclidean on the
standardized alpha metrics; in practice ``adata`` + ``"gunifrac"``):

>>> D = mb.distance_matrix(alpha, standardize=True)
>>> stool.within_distance(D).round(2)
case_id
p1    0.88
p2    0.57
Name: distance_1_2, dtype: float64

With real data the same calls read::

    pairs = mb.find_pairs(adata.obs.join(meta), "case_id", "Visit", [1, 2],
                          by="sample_type", prefer="nreads")
    stool = pairs.subset(sample_type="feces")

    stool.delta(adata, ["Shannon.Effective"], obsm="alpha_diversity")
    stool.delta(adata, layer="raw_counts", transform="clr")   # per-zOTU CLR delta
    stool.wide(adata, "Shannon.Effective", obsm="alpha_diversity")  # for PairedStripBox
    stool.within_distance(adata, "gunifrac")                  # UniFrac V1 vs V2
    shift = stool.shift_distance(adata, "gunifrac")           # patient x patient
    mb.mantel_test(shift, mb.distance_matrix(qol_delta, standardize=True))

:func:`align_samples` re-keys a table that identifies samples by other columns
(e.g. ``mt_id`` + visit) onto the sample IDs, so it can be fed to the methods
above.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .distance import as_distance_frame, distance_matrix, pcoa, subset_distance

__all__ = ["find_pairs", "Pairs", "align_samples", "example_data"]


def example_data():
    """Tiny paired dataset for examples and doctests.

    Three patients (``p1``-``p3``), stool and saliva, visits 1 and 2. ``p3``
    lacks the visit-2 stool sample; everything else is complete.

    Returns
    -------
    meta : DataFrame
        One row per sample (index ``sample_id`` = ``s1`` ... ``s10``) with
        ``case_id``, ``sample_type``, ``Visit``, ``mt_id``.
    alpha : DataFrame
        ``shannon`` and ``richness`` per sample.
    counts : DataFrame
        Raw counts of two OTUs per sample.

    Examples
    --------
    >>> import cbrmb as mb
    >>> meta, alpha, counts = mb.paired.example_data()
    >>> meta.head(3)
              case_id sample_type  Visit mt_id
    sample_id
    s1             p1       feces      1   MT1
    s2             p1       feces      2   MT1
    s3             p2       feces      1   MT2
    """
    idx = pd.Index([f"s{i}" for i in range(1, 11)], name="sample_id")
    meta = pd.DataFrame({
        "case_id": ["p1", "p1", "p2", "p2", "p3", "p1", "p1", "p2", "p3", "p3"],
        "sample_type": ["feces"] * 5 + ["saliva"] * 5,
        "Visit": [1, 2, 1, 2, 1, 1, 2, 1, 1, 2],
        "mt_id": ["MT1", "MT1", "MT2", "MT2", "MT3", "MT1", "MT1", "MT2", "MT3", "MT3"],
    }, index=idx)
    alpha = pd.DataFrame({
        "shannon": [3.0, 2.5, 4.0, 4.4, 3.3, 2.0, 2.6, 2.2, 1.9, 2.4],
        "richness": [120, 95, 160, 171, 130, 60, 75, 66, 58, 70],
    }, index=idx)
    counts = pd.DataFrame({
        "otuA": [10, 30, 5, 5, 8, 1, 2, 3, 4, 5],
        "otuB": [90, 70, 95, 95, 92, 9, 8, 7, 6, 5],
    }, index=idx)
    return meta, alpha, counts


# --- key handling -------------------------------------------------------------

def _as_list(x):
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _norm_key(v):
    """Normalize one key value so 1 == 1.0 == "1" and " ab" == "ab"."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if float(v).is_integer() else repr(float(v))
    if isinstance(v, str):
        s = v.strip()
        try:
            f = float(s)
        except ValueError:
            return s
        return str(int(f)) if f.is_integer() else repr(f)
    return str(v)


def _key_tuples(df, cols):
    keys = [tuple(_norm_key(v) for v in row) for row in zip(*(df[c] for c in cols))]
    return [None if any(p is None for p in k) else k for k in keys]


def _frame_of(meta):
    """``adata.obs`` for AnnData, otherwise ``meta`` itself."""
    if hasattr(meta, "obs") and hasattr(meta, "obs_names"):
        return meta.obs
    return meta


def align_samples(table: pd.DataFrame, meta, on, *, meta_on=None, sample=None,
                  verbose=True) -> pd.DataFrame:
    """Re-index ``table`` by the sample IDs of ``meta``, matching on key columns.

    For tables that identify samples by something other than the sample ID --
    e.g. a covariate sheet keyed by ``mt_id`` and visit -- so that they can be
    passed to :class:`Pairs` methods or to anything expecting sample-indexed data.
    Key values are normalized before matching: ``1``, ``1.0`` and ``"1"`` are
    equal, surrounding whitespace is ignored.

    Parameters
    ----------
    table : DataFrame
        Rows to re-key. Constant parts of the key that ``table`` lacks can be
        added inline, e.g. ``covars.assign(sample_type="feces")``.
    meta : DataFrame or AnnData
        Sample-level metadata (one row per sample). AnnData uses ``.obs``.
    on : str, list of str, or dict
        Key column(s) in ``table``. A dict maps ``table`` column -> ``meta``
        column, e.g. ``{"mt_id_stool": "mt_id", "visit": "Visit",
        "sample_type": "sample_type"}``.
    meta_on : str or list of str, optional
        Matching key columns in ``meta`` when ``on`` is a list and the names
        differ (same order). Defaults to ``on``.
    sample : str, optional
        Column of ``meta`` holding the sample ID; default ``meta.index``.
    verbose : bool
        Print how many rows matched and which did not.

    Returns
    -------
    DataFrame
        The matched rows of ``table`` (all its columns), indexed by sample ID,
        in ``meta`` order. ``.attrs["unmatched"]`` lists the unmatched ``table``
        keys.

    Raises
    ------
    ValueError
        If a key is not unique within ``table`` or within ``meta``.

    Examples
    --------
    A covariate sheet with one row per patient visit, keyed by the stool
    ``mt_id`` and a visit column that came out of Excel half as float, half as
    text. ``sample_type`` is not in the sheet, so it is added as a constant:

    >>> import pandas as pd
    >>> import cbrmb as mb
    >>> meta, alpha, counts = mb.paired.example_data()
    >>> covars = pd.DataFrame({
    ...     "mt_id_stool": ["MT1", "MT1", "MT2", "MT2", "MT9"],
    ...     "visit": [1.0, 2.0, 1.0, "2", 1],
    ...     "antibiotics": [0, 1, 0, 0, 1],
    ... })
    >>> cov = mb.align_samples(
    ...     covars.assign(sample_type="feces"), meta,
    ...     on={"mt_id_stool": "mt_id", "sample_type": "sample_type", "visit": "Visit"})
    align_samples: 4/5 rows matched to a sample; 1 without a sample: MT9/feces/1
    >>> cov[["antibiotics"]]
               antibiotics
    sample_id
    s1                   0
    s2                   1
    s3                   0
    s4                   0

    ``cov`` is now sample-indexed and can go straight into the pair methods:

    >>> pairs = mb.find_pairs(meta, "case_id", "Visit", [1, 2], by="sample_type")
    >>> pairs.subset(sample_type="feces").delta(cov, "antibiotics")
    case_id
    p1    1.0
    p2    0.0
    Name: antibiotics, dtype: float64
    """
    meta = _frame_of(meta)
    if isinstance(on, dict):
        t_cols, m_cols = list(on.keys()), list(on.values())
    else:
        t_cols = _as_list(on)
        m_cols = _as_list(meta_on) if meta_on is not None else t_cols
    if len(t_cols) != len(m_cols):
        raise ValueError(f"on ({t_cols}) and meta_on ({m_cols}) differ in length")

    sample_ids = meta.index if sample is None else pd.Index(meta[sample])
    m_keys = _key_tuples(meta, m_cols)
    t_keys = _key_tuples(table, t_cols)

    m_map = {}
    m_dups = set()
    for k, s in zip(m_keys, sample_ids):
        if k is None:
            continue
        if k in m_map:
            m_dups.add(k)
        m_map[k] = s
    t_seen = pd.Series([k for k in t_keys if k is not None]).value_counts()
    t_dups = list(t_seen.index[t_seen > 1])
    if t_dups:
        raise ValueError(f"{len(t_dups)} key(s) {t_cols} occur more than once in table: {t_dups[:10]}")

    pos, ids, unmatched = [], [], []
    ambiguous = []
    for i, k in enumerate(t_keys):
        if k is None:
            continue
        if k in m_dups:
            ambiguous.append(k)
        elif k in m_map:
            pos.append(i)
            ids.append(m_map[k])
        else:
            unmatched.append(k)
    if ambiguous:
        raise ValueError(f"{len(ambiguous)} key(s) {m_cols} match several samples in meta: {ambiguous[:10]}")

    out = table.iloc[pos].copy()
    out.index = pd.Index(ids, name=sample_ids.name or "sample_id")
    order = sample_ids.get_indexer(out.index)
    out = out.iloc[np.argsort(order, kind="stable")]
    out.attrs["unmatched"] = unmatched

    if verbose:
        n_nan = sum(k is None for k in t_keys)
        msg = f"align_samples: {len(out)}/{len(table)} rows matched to a sample"
        if n_nan:
            msg += f", {n_nan} with an incomplete key"
        if unmatched:
            shown = ", ".join("/".join(k) for k in unmatched[:10])
            more = " ..." if len(unmatched) > 10 else ""
            msg += f"; {len(unmatched)} without a sample: {shown}{more}"
        print(msg)
    return out


# --- data resolution ----------------------------------------------------------

_TRANSFORMS = ("rel", "clr", "log", "log2", "log10")


def _sample_frame(data, cols, *, layer=None, obsm=None) -> pd.DataFrame:
    """Sample-indexed DataFrame from a DataFrame/Series/AnnData."""
    if isinstance(data, pd.Series):
        return data.to_frame(data.name if data.name is not None else "value")
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "obs") and hasattr(data, "obs_names"):
        if obsm is not None:
            m = data.obsm[obsm]
            if isinstance(m, pd.DataFrame):
                return m
            return pd.DataFrame(np.asarray(m), index=data.obs_names)
        if layer is None and cols and all(c in data.obs.columns for c in cols):
            return data.obs
        X = data.layers[layer] if layer is not None else data.X
        return _LazyMatrix(X, data.obs_names, data.var_names)
    raise TypeError(f"cannot use {type(data).__name__} as sample data; pass a DataFrame, Series or AnnData")


class _LazyMatrix:
    """Row-subsettable stand-in for a (possibly sparse) AnnData matrix."""

    def __init__(self, X, index, columns):
        self.X, self.index, self.columns = X, pd.Index(index), pd.Index(columns)

    def take(self, samples):
        pos = self.index.get_indexer(samples)
        X = self.X[pos]
        if hasattr(X, "toarray"):
            X = X.toarray()
        return pd.DataFrame(np.asarray(X), index=pd.Index(samples), columns=self.columns)


def _rows(frame, samples):
    if isinstance(frame, _LazyMatrix):
        return frame.take(samples)
    return frame.loc[samples]


def _apply_transform(df, transform, pseudocount):
    if transform is None:
        return df
    if callable(transform):
        return transform(df)
    if transform not in _TRANSFORMS:
        raise ValueError(f"transform must be one of {_TRANSFORMS}, a callable or None, got {transform!r}")
    X = df.to_numpy(dtype=float)
    if transform == "rel":
        X = X / X.sum(axis=1, keepdims=True)
    elif transform == "clr":
        lx = np.log(X + pseudocount)
        X = lx - lx.mean(axis=1, keepdims=True)
    else:
        X = {"log": np.log, "log2": np.log2, "log10": np.log10}[transform](X + pseudocount)
    return pd.DataFrame(X, index=df.index, columns=df.columns)


# --- Pairs --------------------------------------------------------------------

def _stack(table):
    """Non-missing cells of ``table`` as a Series (index: row levels + visit)."""
    try:
        s = table.stack(future_stack=True)
    except TypeError:  # pandas < 2.1
        s = table.stack(dropna=False)
    return s.dropna()


def find_pairs(meta, unit, visit, visits=None, *, by=None, sample=None, min_visits=None,
               duplicates="warn", prefer=None, ascending=False) -> "Pairs":
    """Find the samples each unit has at each visit.

    Parameters
    ----------
    meta : DataFrame or AnnData
        Sample-level table, one row per sample (AnnData uses ``.obs``).
    unit : str or list of str
        Column(s) identifying the observation unit, e.g. ``"case_id"``.
    visit : str
        Column with the visit / time point.
    visits : sequence, optional
        Visits to line up, in order (``[1, 2]`` for pre/post; labels such as
        ``"baseline"`` work too). Values match by equality, so ``1`` finds
        ``1.0``. Default: all visits present, sorted.
    by : str or list of str, optional
        Column(s) that split units further without being part of the subject
        identity -- typically ``"sample_type"``, so stool and saliva pairs are
        kept apart. They become extra index levels; drop them with
        :meth:`Pairs.subset` or :meth:`Pairs.split`.
    sample : str, optional
        Column with the sample ID; default ``meta.index``.
    min_visits : int, optional
        Keep units with at least this many of ``visits``; default all of them.
        Missing visits are NaN in :attr:`Pairs.table`.
    duplicates : {"warn", "error", "first"}
        What to do when a unit has more than one sample at a visit (re-runs,
        dilutions, ...): keep the first with a warning that lists the clashes,
        raise, or keep the first silently. Passing ``prefer`` counts as an
        explicit rule, so ``"warn"`` stays quiet then.
    prefer : str, optional
        Column to sort duplicates by before keeping the first, e.g.
        ``"nreads"`` to keep the deepest-sequenced sample.
    ascending : bool
        Sort direction for ``prefer`` (default descending: largest wins).

    Examples
    --------
    >>> import cbrmb as mb
    >>> meta, alpha, counts = mb.paired.example_data()

    Pre/post pairs per patient and sample type (``p3`` stool has no visit 2):

    >>> pairs = mb.find_pairs(meta, unit="case_id", visit="Visit", visits=[1, 2],
    ...                       by="sample_type")
    >>> pairs.summary()
    Visit        1  2  complete
    sample_type
    feces        2  2         2
    saliva       2  2         2

    Keep incomplete units too -- the missing visit is NaN:

    >>> mb.find_pairs(meta, "case_id", "Visit", [1, 2], by="sample_type",
    ...               min_visits=1).subset(sample_type="feces").table
    Visit     1    2
    case_id
    p1       s1   s2
    p2       s3   s4
    p3       s5  NaN

    A re-sequenced sample (two samples in one patient-visit slot) is resolved by
    ``prefer``; without it you get a warning listing the clashes:

    >>> meta2 = meta.copy()
    >>> meta2.loc["s1_rerun"] = ["p1", "feces", 1, "MT1"]
    >>> meta2["nreads"] = 10_000
    >>> meta2.loc["s1_rerun", "nreads"] = 25_000
    >>> stool = mb.find_pairs(meta2, "case_id", "Visit", [1, 2], by="sample_type",
    ...                       prefer="nreads").subset(sample_type="feces")
    >>> stool.table.loc["p1"].tolist()
    ['s1_rerun', 's2']

    Any number of visits; ``early``/``late`` on the methods pick which two to
    compare (default first and last)::

        trio = mb.find_pairs(meta, "case_id", "Visit", [1, 2, 3], min_visits=2)
        trio.delta(alpha, early=2, late=3)
    """
    frame = _frame_of(meta)
    unit = _as_list(unit)
    by = _as_list(by)
    if duplicates not in ("warn", "error", "first"):
        raise ValueError(f"duplicates must be 'warn', 'error' or 'first', got {duplicates!r}")

    cols = unit + by + [visit] + ([prefer] if prefer else [])
    sub = frame[[c for c in dict.fromkeys(cols)]].copy()
    sub["_sample"] = frame.index if sample is None else frame[sample].to_numpy()
    sub = sub.dropna(subset=unit + by + [visit, "_sample"])

    if visits is None:
        visits = sorted(sub[visit].unique())
    visits = list(visits)
    if len(set(visits)) != len(visits):
        raise ValueError(f"visits contains duplicates: {visits}")

    label = pd.Series(pd.NA, index=sub.index, dtype=object)
    for v in visits:
        label[(sub[visit] == v).to_numpy()] = v
    sub["_visit"] = label
    sub = sub[sub["_visit"].notna()]

    if prefer:
        sub = sub.sort_values(prefer, ascending=ascending, kind="stable", na_position="last")
    keys = unit + by + ["_visit"]
    dup = sub.duplicated(subset=keys, keep=False)
    if dup.any():
        listing = sub.loc[dup, keys + ["_sample"]].rename(columns={"_visit": visit, "_sample": "sample"})
        listing = listing.sort_values(list(listing.columns[:-1]))
        if duplicates == "error":
            raise ValueError(f"several samples per {unit + by} x {visit}:\n{listing.to_string(index=False)}")
        if duplicates == "warn" and not prefer:
            warnings.warn(f"{dup.sum()} samples share a {unit + by} x {visit} slot - keeping the first "
                          f"(pass prefer= to choose, e.g. prefer='nreads'):\n"
                          f"{listing.to_string(index=False)}", stacklevel=2)
        sub = sub[~sub.duplicated(subset=keys, keep="first")]

    table = sub.pivot(index=unit + by, columns="_visit", values="_sample")
    table = table.reindex(columns=visits)
    table.columns = pd.Index(visits, name=visit)

    need = len(visits) if min_visits is None else int(min_visits)
    table = table[table.notna().sum(axis=1) >= need].sort_index()
    return Pairs(table, unit=unit, by=by, visit=visit)


class Pairs:
    """Samples per unit and visit, plus the operations that consume them.

    :attr:`table` has one row per unit (index levels ``unit`` + ``by``) and one
    column per visit, holding sample IDs (NaN where a visit is missing). All
    methods take *sample-indexed* data -- a DataFrame/Series indexed by sample
    ID, or an AnnData -- and return unit-indexed results that line up with
    each other (and with other unit-indexed tables) by index.

    Methods that compare two visits take ``early``/``late`` (visit labels);
    they default to the first and last visit and use only units that have both.

    Build it with :func:`find_pairs`. Overview of what it gives you:

    ========================  ==========================================  =====================
    method                    result                                      typical use
    ========================  ==========================================  =====================
    :meth:`delta`             unit x feature, ``late - early``            alpha / zOTU / QoL change
    :meth:`wide`              unit x visit values                         paired strip/box plots
    :meth:`within_distance`   unit -> d(early, late)                      "how much did it move"
    :meth:`distance_matrix`   sample x sample sub-matrix                  PERMANOVA on the pairs
    :meth:`long`              sample -> unit, visit                       ``subject=`` covariate
    :meth:`shift_vectors`     unit x PCo, ``late - early`` in PCoA        direction of change
    :meth:`shift_distance`    unit x unit distance of shift vectors       Mantel test
    :meth:`delta_distance`    unit x unit distance of deltas              Mantel test
    ========================  ==========================================  =====================
    """

    def __init__(self, table: pd.DataFrame, *, unit, by=(), visit="visit"):
        self.table = table
        self.unit = list(unit)
        self.by = list(by)
        self.visit = visit

    # -- introspection ---------------------------------------------------------

    @property
    def visits(self) -> list:
        return list(self.table.columns)

    def __len__(self):
        return len(self.table)

    def __repr__(self):
        counts = ", ".join(f"{v}: {n}" for v, n in self.table.notna().sum().items())
        head = f"Pairs({len(self)} units by {self.table.index.names}; samples per {self.visit} -> {counts})"
        if self.by:
            per = self.table.groupby(level=self.by).size()
            head += "\n" + "\n".join(f"  {k}: {n}" for k, n in per.items())
        return head

    def summary(self) -> pd.DataFrame:
        """Units with a sample at each visit, per ``by`` group.

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> mb.find_pairs(meta, "case_id", "Visit", by="sample_type", min_visits=1).summary()
        Visit        1  2  complete
        sample_type
        feces        3  2         2
        saliva       3  2         2
        """
        present = self.table.notna()
        present["complete"] = present.all(axis=1)
        if not self.by:
            return present.sum().to_frame("n").T
        return present.groupby(level=self.by).sum()

    @property
    def samples(self) -> pd.Index:
        """All sample IDs in the pairs (unit order, then visit order)."""
        return pd.Index(_stack(self.table).to_numpy(), name="sample_id")

    def long(self) -> pd.DataFrame:
        """One row per sample: unit/by columns and the visit, indexed by sample ID.

        Handy as ``subject=`` / covariate input next to :meth:`distance_matrix`.

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> stool = mb.find_pairs(meta, "case_id", "Visit", [1, 2],
        ...                       by="sample_type").subset(sample_type="feces")
        >>> stool.long()
                  case_id  Visit
        sample_id
        s1             p1      1
        s2             p1      2
        s3             p2      1
        s4             p2      2

        Repeated-measures PERMANOVA restricted to the paired samples::

            d = stool.distance_matrix(adata, "gunifrac")
            mb.screen_confounder(d, covars.loc[d.index], subject=stool.long()["case_id"])
        """
        s = _stack(self.table)
        out = s.index.to_frame(index=False)
        out.index = pd.Index(s.to_numpy(), name="sample_id")
        return out

    # -- restructuring ----------------------------------------------------------

    def _new(self, table, by=None):
        return Pairs(table, unit=self.unit, by=self.by if by is None else by, visit=self.visit)

    def subset(self, **levels) -> "Pairs":
        """Keep the units matching ``by`` levels, dropping those index levels.

        ``pairs.subset(sample_type="feces")`` -> index is just the unit.
        A list value keeps several levels (and the index level stays).

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> pairs = mb.find_pairs(meta, "case_id", "Visit", [1, 2], by="sample_type")
        >>> pairs.subset(sample_type="saliva").table
        Visit     1    2
        case_id
        p1       s6   s7
        p3       s9  s10
        """
        t = self.table
        keep_by = list(self.by)
        for name, value in levels.items():
            if name not in self.by:
                raise KeyError(f"{name!r} is not a by-level of these pairs ({self.by})")
            vals = t.index.get_level_values(name)
            if isinstance(value, (list, tuple, set, pd.Index, np.ndarray)):
                t = t[vals.isin(list(value))]
            else:
                t = t[vals == value]
                t = t.droplevel(name)
                keep_by.remove(name)
        return self._new(t, by=keep_by)

    def split(self, level=None) -> dict:
        """``{value: Pairs}`` per value of a ``by`` level (default: the only one).

        Loop over sample types instead of copy-pasting the stool cell for saliva:

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> pairs = mb.find_pairs(meta, "case_id", "Visit", [1, 2], by="sample_type")
        >>> {st: round(float(p.delta(alpha, "shannon").mean()), 2) for st, p in pairs.split().items()}
        {'feces': -0.05, 'saliva': 0.55}
        """
        if level is None:
            if len(self.by) != 1:
                raise ValueError(f"pass level= (one of {self.by})")
            level = self.by[0]
        values = self.table.index.get_level_values(level).unique()
        return {v: self.subset(**{level: v}) for v in values}

    def complete(self, early=None, late=None) -> pd.DataFrame:
        """Units with a sample at both ``early`` and ``late``: two columns of sample IDs."""
        early, late = self._ends(early, late)
        return self.table[[early, late]].dropna()

    def _ends(self, early, late):
        visits = self.visits
        early = visits[0] if early is None else early
        late = visits[-1] if late is None else late
        for v in (early, late):
            if v not in visits:
                raise KeyError(f"visit {v!r} not in {visits}")
        if early == late:
            raise ValueError("early and late are the same visit")
        return early, late

    # -- values -----------------------------------------------------------------

    def _values(self, data, cols, *, layer, obsm, transform, pseudocount, samples):
        cols_list = _as_list(cols) if cols is not None else None
        frame = _sample_frame(data, cols_list, layer=layer, obsm=obsm)
        samples = pd.Index(samples)
        present = samples.isin(frame.index)
        if not present.all():
            missing = list(samples[~present])
            warnings.warn(f"{len(missing)} paired sample(s) missing from data, their units are dropped: "
                          f"{missing[:10]}", stacklevel=3)
        vals = _rows(frame, samples[present])
        vals = _apply_transform(vals, transform, pseudocount)
        if cols_list is not None:
            vals = vals[cols_list]
        return vals

    def delta(self, data, cols=None, *, early=None, late=None, transform=None, pseudocount=1.0,
              layer=None, obsm=None):
        """``late - early`` per unit.

        Parameters
        ----------
        data : DataFrame, Series or AnnData
            Sample-indexed values: alpha diversity, a count/abundance table,
            clinical covariates, ... For AnnData: ``obsm=`` picks an ``obsm``
            table, columns of ``.obs`` are used when all ``cols`` are there,
            otherwise ``.X`` / ``layers[layer]`` (features = ``var_names``).
        cols : str or list of str, optional
            Columns to difference; default all. A single string returns a Series.
        transform : {"rel", "clr", "log", "log2", "log10"} or callable, optional
            Applied per sample to the *full* table before ``cols`` are picked
            (so CLR uses all features). ``"log2"`` gives log2 fold changes.
            A callable gets and returns the sample-indexed DataFrame.
        pseudocount : float
            Added before ``clr`` / ``log*``.

        Returns
        -------
        DataFrame (or Series for a single ``cols`` string), indexed like
        :attr:`table`, only units with both visits present in ``data``.

        Examples
        --------
        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> stool = mb.find_pairs(meta, "case_id", "Visit", [1, 2],
        ...                       by="sample_type").subset(sample_type="feces")
        >>> stool.delta(alpha)
                 shannon  richness
        case_id
        p1          -0.5     -25.0
        p2           0.4      11.0

        Abundance changes: relative abundance, or log2 fold change per OTU:

        >>> stool.delta(counts, transform="rel").round(2)
                 otuA  otuB
        case_id
        p1        0.2  -0.2
        p2        0.0   0.0
        >>> stool.delta(counts, "otuA", transform="log2").round(2)
        case_id
        p1    1.49
        p2    0.00
        Name: otuA, dtype: float64

        The result is a plain unit-indexed table, so it joins with other
        per-patient tables and feeds the per-feature tests directly::

            d_alpha = stool.delta(adata, ["Shannon.Effective", "Richness"], obsm="alpha_diversity")
            d_zotu = stool.delta(adata, layer="raw_counts", transform="clr")
            mb.corr_test(qol_delta.loc[d_alpha.index], d_alpha["Shannon.Effective"])
        """
        early, late = self._ends(early, late)
        pairs = self.complete(early, late)
        vals = self._values(data, cols, layer=layer, obsm=obsm, transform=transform,
                            pseudocount=pseudocount, samples=pd.unique(pairs.to_numpy().ravel()))
        ok = pairs[early].isin(vals.index) & pairs[late].isin(vals.index)
        pairs = pairs[ok]
        a = vals.loc[pairs[early]].to_numpy(dtype=float)
        b = vals.loc[pairs[late]].to_numpy(dtype=float)
        out = pd.DataFrame(b - a, index=pairs.index, columns=vals.columns)
        out.attrs["delta"] = f"{late} - {early}"
        if isinstance(cols, str):
            return out[cols]
        return out

    def wide(self, data, cols=None, *, transform=None, pseudocount=1.0, layer=None, obsm=None):
        """Values at each visit side by side, per unit.

        One ``cols`` string -> unit x visit DataFrame (ready for
        ``cbrviz.PairedStripBox``); several -> columns ``(col, visit)``.
        Missing visits / samples are NaN.

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> stool = mb.find_pairs(meta, "case_id", "Visit", [1, 2],
        ...                       by="sample_type").subset(sample_type="feces")
        >>> stool.wide(alpha, "shannon")
        Visit      1    2
        case_id
        p1       3.0  2.5
        p2       4.0  4.4

        Straight into a paired plot::

            sb = viz.PairedStripBox(stool.wide(adata, "Shannon.Effective", obsm="alpha_diversity")
                                    .set_axis(["Visit 1", "Visit 2"], axis=1))
        """
        long = _stack(self.table)
        vals = self._values(data, cols, layer=layer, obsm=obsm, transform=transform,
                            pseudocount=pseudocount, samples=pd.unique(long.to_numpy()))
        long = long[long.isin(vals.index)]
        stacked = pd.DataFrame(vals.loc[long.to_numpy()].to_numpy(), index=long.index, columns=vals.columns)
        out = stacked.unstack(self.visit)
        out = out.reindex(columns=pd.MultiIndex.from_product([vals.columns, self.visits],
                                                             names=[None, self.visit]))
        out = out.reindex(self.table.index)
        if isinstance(cols, str):
            return out[cols]
        return out

    def delta_distance(self, data, cols=None, *, metric="euclidean", standardize=False, **delta_kwargs):
        """Unit x unit distance matrix of the deltas (see :meth:`delta`).

        E.g. several QoL domain deltas with ``standardize=True``, or one binary
        covariate with ``metric="cityblock"`` (= ``|delta_i - delta_j|``).

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> stool = mb.find_pairs(meta, "case_id", "Visit", [1, 2],
        ...                       by="sample_type").subset(sample_type="feces")
        >>> stool.delta_distance(alpha, "shannon", metric="cityblock")
        case_id   p1   p2
        case_id
        p1       0.0  0.9
        p2       0.9  0.0

        Typical: antibiotic start/stop (0/1 per visit) vs. microbiome shift::

            abx = stool.delta_distance(cov, "antibiotics", metric="cityblock")
            mb.mantel_test(stool.shift_distance(adata, "gunifrac"), abx)
        """
        d = self.delta(data, cols, **delta_kwargs)
        return distance_matrix(d, metric=metric, standardize=standardize)

    # -- distances ---------------------------------------------------------------

    def distance_matrix(self, dist, key=None, *, labels=None) -> pd.DataFrame:
        """Sample x sample sub-matrix of ``dist`` for all paired samples.

        Pair it with :meth:`long` for repeated-measures PERMANOVA input.
        ``dist`` is an AnnData (``key`` = ``obsp`` slot), a labelled DataFrame,
        or an ndarray with ``labels=``.

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> stool = mb.find_pairs(meta, "case_id", "Visit", [1, 2],
        ...                       by="sample_type").subset(sample_type="feces")
        >>> D = mb.distance_matrix(alpha, standardize=True)     # stands in for UniFrac
        >>> stool.distance_matrix(D).round(2)
        sample_id    s1    s2    s3    s4
        sample_id
        s1         0.00  0.88  1.60  2.17
        s2         0.88  0.00  2.48  3.04
        s3         1.60  2.48  0.00  0.57
        s4         2.17  3.04  0.57  0.00

        With AnnData: ``stool.distance_matrix(adata, "gunifrac")``.
        """
        return subset_distance(dist, self.samples, key, labels=labels)

    def within_distance(self, dist, key=None, *, labels=None, early=None, late=None) -> pd.Series:
        """Distance between each unit's ``early`` and ``late`` sample.

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> pairs = mb.find_pairs(meta, "case_id", "Visit", [1, 2], by="sample_type")
        >>> D = mb.distance_matrix(alpha, standardize=True)     # stands in for UniFrac
        >>> pairs.within_distance(D).round(2)
        case_id  sample_type
        p1       feces          0.88
                 saliva         0.84
        p2       feces          0.57
        p3       saliva         0.69
        Name: distance_1_2, dtype: float64

        With AnnData the Series is named after the ``obsp`` key, ready to
        correlate with per-patient changes::

            uf = stool.within_distance(adata, "gunifrac")     # name: gunifrac_1_2
            mb.corr_test(qol_delta.loc[uf.index], uf)
        """
        early, late = self._ends(early, late)
        pairs = self.complete(early, late)
        d = as_distance_frame(dist, key, labels=labels)
        i = d.index.get_indexer(pairs[early])
        j = d.index.get_indexer(pairs[late])
        ok = (i >= 0) & (j >= 0)
        if not ok.all():
            missing = sorted(set(pairs[early][i < 0]) | set(pairs[late][j < 0]))
            warnings.warn(f"{len(missing)} paired sample(s) not in the distance matrix, their units are "
                          f"dropped: {missing[:10]}", stacklevel=2)
        return pd.Series(d.to_numpy()[i[ok], j[ok]], index=pairs.index[ok],
                         name=f"{key or 'distance'}_{early}_{late}")

    def shift_vectors(self, dist, key=None, *, labels=None, early=None, late=None,
                      n_axes=None) -> pd.DataFrame:
        """Per-unit shift ``late - early`` in PCoA space.

        Runs a classical PCoA (:func:`cbrmb.distance.pcoa`) on the sub-matrix of
        the units' ``early`` and ``late`` samples, then subtracts coordinates.
        Subset to one ``by`` level (e.g. one sample type) first unless you
        really want them in one ordination. ``.attrs["explained_variance_ratio"]``
        is carried over from the PCoA.

        For a Euclidean ``dist`` the shift vectors are just the deltas of the
        underlying features, rotated -- so their distances agree:

        >>> import numpy as np
        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> saliva = mb.find_pairs(meta, "case_id", "Visit", [1, 2],
        ...                        by="sample_type").subset(sample_type="saliva")
        >>> vec = saliva.shift_vectors(mb.distance_matrix(alpha))
        >>> bool(np.isclose(np.linalg.norm(vec.loc["p1"]),
        ...                 np.linalg.norm(saliva.delta(alpha).loc["p1"])))
        True
        """
        early, late = self._ends(early, late)
        pairs = self.complete(early, late)
        sub = subset_distance(dist, pd.unique(pairs.to_numpy().ravel()), key, labels=labels)
        coords = pcoa(sub, n_axes=n_axes)
        out = pd.DataFrame(
            coords.loc[pairs[late]].to_numpy() - coords.loc[pairs[early]].to_numpy(),
            index=pairs.index, columns=coords.columns,
        )
        out.attrs = dict(coords.attrs)
        return out

    def shift_distance(self, dist, key=None, *, labels=None, early=None, late=None, n_axes=None,
                       metric="euclidean") -> pd.DataFrame:
        """Unit x unit distance between microbiome shift vectors (Mantel input).

        Two patients are close when their microbiome moved the same way between
        the visits (not merely by the same amount -- that is
        :meth:`within_distance`).

        >>> import cbrmb as mb
        >>> meta, alpha, counts = mb.paired.example_data()
        >>> pairs = mb.find_pairs(meta, "case_id", "Visit", [1, 2], by="sample_type")
        >>> saliva = pairs.subset(sample_type="saliva")
        >>> saliva.shift_distance(mb.distance_matrix(alpha, standardize=True)).round(2)
        case_id    p1    p3
        case_id
        p1       0.00  0.15
        p3       0.15  0.00

        The whole Mantel recipe, per sample type and between them::

            shift = {st: p.shift_distance(adata, "gunifrac") for st, p in pairs.split().items()}
            mb.mantel_test(shift["feces"], qol_dist)
            mb.mantel_test(shift["feces"], shift["saliva"])     # shared patients only
        """
        vec = self.shift_vectors(dist, key, labels=labels, early=early, late=late, n_axes=n_axes)
        return distance_matrix(vec, metric=metric)
