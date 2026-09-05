"""Generalized UniFrac distances in pure Python (no R, no rpy2).

Port of the ``d_alpha`` family from R's ``GUniFrac::GUniFrac`` (Chen et al.,
*Bioinformatics* 2012), plus a small Newick reader and a midpoint-rooting
routine equivalent to ``phangorn::midpoint``.  Only NumPy / pandas are needed.

The original notebook recipe was::

    tree <- read.tree("ZOTUs-Tree-nj.tre")
    tree$tip.label <- gsub("'", "", tree$tip.label)
    rooted_tree <- phangorn::midpoint(tree)
    unifracs <- GUniFrac(t(norm_counts), rooted_tree, alpha = c(0, 0.5, 1))$unifracs
    udist <- unifracs[, , "d_0.5"]

which becomes::

    from cbrmb.unifrac import calc_gunifrac
    udist = calc_gunifrac(norm_counts, "ZOTUs-Tree-nj.tre", alpha=0.5)

``norm_counts`` here is samples (rows) x OTUs (columns) -- the same orientation
the rest of :mod:`cbrmb` uses, i.e. no transpose.

For a given branch ``b`` with length ``l_b`` and the fraction ``p_b`` of each
community that sits below it, the distance between samples ``i`` and ``j`` is

    d_alpha(i, j) = sum_b l_b (p_bi + p_bj)^alpha |p_bi - p_bj| / (p_bi + p_bj)
                    -------------------------------------------------------------
                              sum_b l_b (p_bi + p_bj)^alpha

summed over branches with ``p_bi + p_bj > 0``.  ``alpha=1`` is weighted
normalized UniFrac; ``alpha=0`` weights every branch equally regardless of
abundance; ``alpha=0.5`` is the usual compromise with the best power in the
Chen et al. simulations.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["generalized_unifrac", "calc_gunifrac", "read_newick", "root_at_midpoint"]


# --------------------------------------------------------------------------- #
# Minimal tree structure + Newick parser
# --------------------------------------------------------------------------- #
class _Node:
    """A rooted-tree node.  ``length`` is the branch above it (to ``parent``)."""

    __slots__ = ("name", "length", "children", "parent")

    def __init__(self, name=None, length=None):
        self.name = name
        self.length = length
        self.children: list["_Node"] = []
        self.parent: "_Node | None" = None

    def is_tip(self) -> bool:
        return not self.children


def _strip_comments(s: str) -> str:
    """Drop ``[...]`` Newick/Nexus comments that sit outside quoted labels."""
    out, i, n, in_quote = [], 0, len(s), False
    while i < n:
        c = s[i]
        if in_quote:
            out.append(c)
            if c == "'":
                if i + 1 < n and s[i + 1] == "'":
                    out.append("'")
                    i += 2
                    continue
                in_quote = False
            i += 1
        elif c == "'":
            in_quote = True
            out.append(c)
            i += 1
        elif c == "[":
            depth = 1
            i += 1
            while i < n and depth:
                if s[i] == "[":
                    depth += 1
                elif s[i] == "]":
                    depth -= 1
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _parse_newick(text: str) -> _Node:
    s = _strip_comments(text).strip()
    if not s:
        raise ValueError("empty Newick string")
    pos = 0
    n = len(s)

    def skip_ws():
        nonlocal pos
        while pos < n and s[pos].isspace():
            pos += 1

    def read_label() -> "str | None":
        nonlocal pos
        skip_ws()
        if pos < n and s[pos] == "'":
            pos += 1
            buf = []
            while pos < n:
                if s[pos] == "'":
                    if pos + 1 < n and s[pos + 1] == "'":
                        buf.append("'")
                        pos += 2
                        continue
                    pos += 1
                    break
                buf.append(s[pos])
                pos += 1
            return "".join(buf)
        start = pos
        while pos < n and s[pos] not in ":,()[];":
            pos += 1
        lab = s[start:pos].strip()
        return lab or None

    def read_number() -> "float | None":
        nonlocal pos
        skip_ws()
        start = pos
        while pos < n and (s[pos].isdigit() or s[pos] in "+-.eE"):
            pos += 1
        chunk = s[start:pos].strip()
        return float(chunk) if chunk else None

    def parse_clade() -> _Node:
        nonlocal pos
        skip_ws()
        node = _Node()
        if pos < n and s[pos] == "(":
            pos += 1
            while True:
                child = parse_clade()
                child.parent = node
                node.children.append(child)
                skip_ws()
                if pos >= n:
                    raise ValueError("unbalanced parentheses in Newick string")
                if s[pos] == ",":
                    pos += 1
                    continue
                if s[pos] == ")":
                    pos += 1
                    break
                raise ValueError(f"unexpected {s[pos]!r} at position {pos}")
        node.name = read_label()
        skip_ws()
        if pos < n and s[pos] == ":":
            pos += 1
            node.length = read_number()
        return node

    root = parse_clade()
    skip_ws()
    if pos < n and s[pos] == ";":
        pos += 1
    return root


def read_newick(source, *, strip_tip_quotes: bool = True) -> _Node:
    """Parse a Newick tree from a path, open file, or string; return the root.

    ``strip_tip_quotes`` additionally removes any residual single quotes from
    tip labels (the parser already unquotes ``'...'`` labels), mirroring the
    ``gsub("'", "", tree$tip.label)`` line in the R recipe.
    """
    if hasattr(source, "read"):
        text = source.read()
    else:
        text = str(source)
        if "(" not in text and ";" not in text:  # looks like a path
            with open(source, "r") as fh:
                text = fh.read()
    root = _parse_newick(text)
    if strip_tip_quotes:
        for node in _iter_postorder(root):
            if node.is_tip() and node.name:
                node.name = node.name.replace("'", "")
    return root


# --------------------------------------------------------------------------- #
# Traversal helpers
# --------------------------------------------------------------------------- #
def _iter_postorder(root: _Node):
    """Yield nodes children-before-parents, iteratively (no recursion limit)."""
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            yield node
        else:
            stack.append((node, True))
            for child in node.children:
                stack.append((child, False))


# --------------------------------------------------------------------------- #
# Midpoint rooting  (equivalent to phangorn::midpoint)
# --------------------------------------------------------------------------- #
def _unrooted_adjacency(root: _Node):
    """{node_key: {neighbour_key: branch_length}} over the whole tree."""
    adj: dict = {}
    id2node: dict = {}
    for node in _iter_postorder(root):
        key = id(node)
        id2node[key] = node
        adj.setdefault(key, {})
        for child in node.children:
            w = float(child.length) if child.length is not None else 0.0
            adj[key][id(child)] = w
            adj.setdefault(id(child), {})[key] = w
    return adj, id2node


def _farthest(adj: dict, start):
    """Farthest node from ``start`` in the (edge-weighted) tree + predecessors."""
    best_d, best_n = 0.0, start
    prev = {start: None}
    stack = [(start, 0.0)]
    while stack:
        u, d = stack.pop()
        if d > best_d:
            best_d, best_n = d, u
        for v, w in adj[u].items():
            if v not in prev:
                prev[v] = u
                stack.append((v, d + max(w, 0.0)))
    return best_d, best_n, prev


def root_at_midpoint(root: _Node) -> _Node:
    """Return a new tree rooted at the midpoint of its longest tip-to-tip path.

    Matches ``phangorn::midpoint`` on the common case (a single longest path,
    non-negative branch lengths).  Negative branch lengths -- occasionally
    produced by neighbour-joining -- are treated as zero while locating the
    path, but kept as-is in the returned tree and in the distance computation.
    """
    adj, id2node = _unrooted_adjacency(root)
    if len(adj) < 3:
        return read_newick(_write_newick(root))  # trivial tree: hand back a copy

    any_key = next(iter(adj))
    _, u, _ = _farthest(adj, any_key)
    diameter, v, prev = _farthest(adj, u)
    if diameter <= 0:
        return read_newick(_write_newick(root))

    # path u -> ... -> v
    path = [v]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()

    half = diameter / 2.0
    cum = 0.0
    a = b = None
    left = right = 0.0
    for k in range(len(path) - 1):
        a, b = path[k], path[k + 1]
        w = adj[a][b]
        if cum + w >= half or k == len(path) - 2:
            left = max(half - cum, 0.0)          # root sits `left` past `a`
            right = max(w - left, 0.0)
            break
        cum += w

    new_key = "__midpoint_root__"
    adj[a].pop(b, None)
    adj[b].pop(a, None)
    adj[new_key] = {a: left, b: right}
    adj[a][new_key] = left
    adj[b][new_key] = right

    return _rebuild_rooted(adj, id2node, new_key)


def _rebuild_rooted(adj: dict, id2node: dict, root_key) -> _Node:
    """Orient the (now split) graph away from ``root_key`` into a rooted tree."""
    root = _Node()
    made = {root_key: root}
    stack = [(root_key, None)]
    while stack:
        cur, parent = stack.pop()
        nd = made[cur]
        for nb, w in adj[cur].items():
            if nb == parent:
                continue
            src = id2node.get(nb)
            child = _Node(name=None if src is None else src.name, length=float(w))
            child.parent = nd
            nd.children.append(child)
            made[nb] = child
            stack.append((nb, cur))
    return root


def _shear(root: _Node, keep) -> _Node:
    """Prune the tree to ``keep`` tips, like ``ape::drop.tip``.

    Internal nodes left with a single kept child are collapsed and their branch
    lengths summed; everything above the MRCA of the kept tips (the "root
    stalk") is discarded.  This matches what ``GUniFrac`` does internally and is
    what makes near-root branches (where every observed community has 100% of
    its mass below them) drop out of the normalisation, exactly as in R.
    """
    keep = set(keep)
    post = list(_iter_postorder(root))
    kc: dict = {}
    for node in post:
        kc[id(node)] = (
            (1 if node.name in keep else 0)
            if node.is_tip()
            else sum(kc[id(c)] for c in node.children)
        )
    if kc[id(root)] < 2:
        raise ValueError("fewer than two of the tree's tips are present in `counts`")

    def is_keeper(node: _Node) -> bool:
        if kc[id(node)] == 0:
            return False
        if node.is_tip():
            return True
        return sum(1 for c in node.children if kc[id(c)] > 0) >= 2

    new = {id(n): _Node(name=n.name) for n in post if is_keeper(n)}
    new_root = None
    for node in post:
        if id(node) not in new:
            continue
        length = node.length or 0.0
        anc = node.parent
        while anc is not None and id(anc) not in new:
            length += anc.length or 0.0
            anc = anc.parent
        child = new[id(node)]
        if anc is None:
            child.length = None
            new_root = child
        else:
            child.length = length
            child.parent = new[id(anc)]
            new[id(anc)].children.append(child)
    return new_root


def _write_newick(node: _Node) -> str:
    """Small Newick writer (used only for cheap deep copies of trivial trees)."""
    def enc(n: _Node) -> str:
        inner = ""
        if n.children:
            inner = "(" + ",".join(enc(c) for c in n.children) + ")"
        lab = "" if not n.name else n.name
        blen = "" if n.length is None else f":{n.length}"
        return f"{inner}{lab}{blen}"

    return enc(node) + ";"


# --------------------------------------------------------------------------- #
# Generalized UniFrac
# --------------------------------------------------------------------------- #
def _branch_table(root: _Node, tip_matrix: np.ndarray, tip_pos: dict):
    """Accumulate community fractions along every branch (post-order sweep)."""
    n_samples = tip_matrix.shape[0]
    abund: dict = {}
    lengths: list[float] = []
    vectors: list[np.ndarray] = []
    for node in _iter_postorder(root):
        if node.is_tip():
            col = tip_pos.get(node.name)
            a = (
                tip_matrix[:, col].copy()
                if col is not None
                else np.zeros(n_samples)
            )
        else:
            a = np.zeros(n_samples)
            for child in node.children:
                a += abund.pop(id(child))
        abund[id(node)] = a
        if node.parent is not None and node.length is not None:
            lengths.append(float(node.length))
            vectors.append(a)
    if not vectors:
        raise ValueError("tree has no usable branches (missing branch lengths?)")
    return np.asarray(lengths), np.vstack(vectors)


def generalized_unifrac(
    counts: pd.DataFrame,
    tree,
    *,
    alphas: Iterable[float] = (0.0, 0.5, 1.0),
    midpoint: bool = True,
    strip_tip_quotes: bool = True,
    validate: bool = True,
) -> dict:
    """Generalized UniFrac distance matrices, one per ``alpha``.

    Parameters
    ----------
    counts : DataFrame
        Samples in rows, OTUs in columns.  Column names must match tree tip
        labels.  Rows are renormalized to sum to 1 (already-normalized input is
        fine -- it is idempotent), matching ``GUniFrac``'s internal step.
    tree : str, path, file, or parsed root node
        Newick tree, or a node returned by :func:`read_newick` /
        :func:`root_at_midpoint`.
    alphas : iterable of float
        Weighting exponents.  ``0.5`` reproduces ``d_0.5`` from the recipe.
    midpoint : bool
        Midpoint-root the tree first (the recipe does; ``GUniFrac`` requires a
        rooted tree).  Only used when ``tree`` needs parsing; pass an already
        rooted node to skip it.  The tree is then pruned to the OTUs in
        ``counts`` (like ``GUniFrac``'s internal ``drop.tip``).
    strip_tip_quotes : bool
        Passed to :func:`read_newick` when ``tree`` needs parsing.
    validate : bool
        Raise if ``counts`` has OTUs absent from the tree (matches ``GUniFrac``
        1.9).  With ``False`` such columns are dropped after normalization (they
        still count toward each row's total, as in R).

    Returns
    -------
    dict[float, DataFrame]
        ``{alpha: square DataFrame}`` indexed and columned by ``counts.index``.
    """
    if not isinstance(counts, pd.DataFrame):
        raise TypeError("counts must be a DataFrame (samples x OTUs)")
    alphas = tuple(float(a) for a in alphas)

    if isinstance(tree, _Node):
        root = tree
    else:
        root = read_newick(tree, strip_tip_quotes=strip_tip_quotes)
        if midpoint:
            root = root_at_midpoint(root)

    tip_names = [n.name for n in _iter_postorder(root) if n.is_tip()]
    if len(tip_names) != len(set(tip_names)):
        raise ValueError("tree has duplicate tip labels")
    tip_set = set(tip_names)

    cols = list(counts.columns)
    unknown = [c for c in cols if c not in tip_set]
    if unknown:
        msg = (
            f"{len(unknown)} of {len(cols)} OTUs in `counts` are not tree tips "
            f"(e.g. {unknown[:5]})"
        )
        if validate:
            raise ValueError(msg + "; pass validate=False to drop them")
        warnings.warn(msg + "; dropping them after normalization", stacklevel=2)

    X = counts.to_numpy(dtype=float)
    row_sum = X.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    X = X / row_sum

    col_pos = {c: k for k, c in enumerate(cols)}
    used = [name for name in tip_names if name in col_pos]
    if len(used) < 2:
        raise ValueError("fewer than two `counts` OTUs are present in the tree")

    # Prune the tree to the observed OTUs, exactly like GUniFrac's internal
    # drop.tip.  Without this, near-root branches that carry the whole observed
    # community (p == 1) stay in the denominator and bias every distance.
    if len(used) < len(tip_set):
        root = _shear(root, used)

    tip_pos = {
        n.name: k
        for k, n in enumerate(m for m in _iter_postorder(root) if m.is_tip())
    }
    tip_matrix = np.zeros((len(counts), len(tip_pos)))
    for name, k in tip_pos.items():
        tip_matrix[:, k] = X[:, col_pos[name]]

    lengths, branch = _branch_table(root, tip_matrix, tip_pos)  # (B,), (B, N)

    n = len(counts)
    mats = {a: np.zeros((n, n)) for a in alphas}
    block = 512
    for i in range(n - 1):
        pi = branch[:, i][:, None]                       # (B, 1)
        for j0 in range(i + 1, n, block):
            j1 = min(j0 + block, n)
            pj = branch[:, j0:j1]                        # (B, m)
            s = pi + pj
            mask = s > 0
            diff = np.abs(pi - pj)
            ratio = np.zeros_like(s)
            np.divide(diff, s, out=ratio, where=mask)
            for a in alphas:
                if a == 0.0:
                    sa = mask.astype(float)
                else:
                    sa = np.where(mask, s, 1.0) ** a
                    sa[~mask] = 0.0
                w = lengths[:, None] * sa
                den = w.sum(axis=0)
                num = (w * ratio).sum(axis=0)
                d = np.divide(num, den, out=np.zeros_like(den), where=den > 0)
                mats[a][i, j0:j1] = d
                mats[a][j0:j1, i] = d

    idx = counts.index
    return {a: pd.DataFrame(m, index=idx, columns=idx) for a, m in mats.items()}


def calc_gunifrac(
    counts: pd.DataFrame,
    tree_path,
    *,
    alpha=0.5,
    midpoint: bool = True,
    strip_tip_quotes: bool = True,
    validate: bool = True,
):
    """Generalized UniFrac, drop-in for the old R-backed ``calc_gunifrac``.

    ``alpha`` may be a scalar (returns one square DataFrame) or an iterable
    (returns ``{alpha: DataFrame}``).  See :func:`generalized_unifrac`.
    """
    scalar = np.isscalar(alpha)
    alphas = (float(alpha),) if scalar else tuple(float(a) for a in alpha)
    res = generalized_unifrac(
        counts,
        tree_path,
        alphas=alphas,
        midpoint=midpoint,
        strip_tip_quotes=strip_tip_quotes,
        validate=validate,
    )
    return res[alphas[0]] if scalar else res
