"""Plotting helpers (sequencing-depth diagnostics, phylum legends).

Needs the ``plotting`` extra (``pip install cbrmb[plotting]``); the matplotlib
import is deferred to call time so ``import cbrmb`` works without it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plot_read_depth", "phylum_handles"]


def plot_read_depth(
    depths,
    cutoff=5000,
    *,
    step=500,
    xmax=None,
    ax=None,
    inset=True,
    inset_bounds=(0.5, 0.2, 0.3, 0.35),
    inset_xlim=None,
    inset_ylim=None,
    color="0.5",
    cutoff_color="r",
    title=None,
    figsize=(3, 3),
    verbose=False,
):
    """ECDF of per-sample sequencing depth, with a cutoff called out.

    Plots the fraction of samples at or below each read count as a filled
    step curve and marks ``cutoff`` with a guide line. By default also draws
    a zoomed-in inset around the cutoff, so a low-depth tail stays visible
    even when it is a small fraction of samples.

    Parameters
    ----------
    depths : array-like
        Per-sample total read counts, e.g. ``counts.sum(axis=1)`` for a
        samples x features table, or ``counts.sum()`` for a features x
        samples table.
    cutoff : float
        Read-depth threshold to highlight.
    step : float
        Grid spacing for the ECDF.
    xmax : float, optional
        Upper bound of the grid; defaults to the smallest multiple of
        ``step`` above ``max(depths)``.
    ax : matplotlib Axes, optional
        Draw into this axes instead of creating a new figure.
    inset : bool
        Add a zoomed inset around ``cutoff``.
    inset_bounds : tuple
        ``[x0, y0, width, height]`` for :meth:`Axes.inset_axes`, in axes
        fraction.
    inset_xlim, inset_ylim : tuple, optional
        Override the inset's default zoom window (``0`` to ``2 * cutoff``;
        ``0`` to ``1.5x`` the fraction of samples at or below ``cutoff``).
    color, cutoff_color : str
        Fill color and guide-line color.
    title : str, optional
        Axes title; defaults to ``"n=<len(depths)>"``.
    figsize : tuple
        Passed to ``plt.subplots`` when ``ax`` is not given.
    verbose : bool
        Print the fraction of samples at or below ``cutoff``.

    Returns
    -------
    fig, ax : the parent figure (``None`` if ``ax`` was passed in) and the
        main axes.
    """
    import matplotlib.pyplot as plt

    depths = np.asarray(depths)
    if xmax is None:
        xmax = step * (int(depths.max() // step) + 2)
    grid = np.arange(0, xmax, step)
    ys = np.array([(depths <= i).mean() for i in grid])
    cut_frac = (depths <= cutoff).mean()

    if verbose:
        print(f"{cut_frac:.1%} of samples have <= {cutoff} reads.")

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(grid, ys, lw=3, facecolor=color, alpha=0.5)
    ax.set_ylabel("Fraction of samples")
    ax.set_xlabel("Reads detected")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], [f"{i:.0%}" for i in [0, 0.25, 0.5, 0.75, 1]])
    ax.set_title(title if title is not None else f"n={len(depths)}", fontsize="small")
    ax.plot([cutoff, cutoff, 0], [0, cut_frac, cut_frac], c=cutoff_color, lw=0.5)

    if inset:
        axin = ax.inset_axes(inset_bounds)
        axin.fill_between(grid, ys, lw=2, facecolor=color)
        axin.set_yticks([0, cut_frac], [f"{i:.0%}" for i in [0, cut_frac]])
        axin.set_xlim(*(inset_xlim or (0, cutoff * 2)))
        axin.set_ylim(*(inset_ylim or (0, max(cut_frac * 1.5, 1e-3))))
        axin.plot([cutoff, cutoff, 0], [0, cut_frac, cut_frac], c=cutoff_color, lw=0.5)
        ax.indicate_inset_zoom(axin, edgecolor="black")

    return fig, ax


def phylum_handles(names, *, colors=None, marker="s", markersize=6, **line2d_kwargs):
    """Legend handles (one ``Line2D`` marker per label) for a phylum bar chart.

    ``names`` are used verbatim as labels; colours come from ``colors`` (a list
    aligned to ``names``, or a name -> colour mapping) or, by default,
    :func:`cbrmb.palettes.phylum_colors`. Pair with ``cbrviz.CompBars`` like::

        cb = CompBars(dsub, colors=mb.phylum_colors(dsub.columns))
        cb.draw(ax=ax)
        fig.legend(handles=mb.phylum_handles(dsub.columns))
    """
    from matplotlib.lines import Line2D

    from .palettes import phylum_colors

    names = list(names)
    if colors is None:
        seq = phylum_colors(names)
    elif isinstance(colors, dict):
        seq = [colors[n] for n in names]
    else:
        seq = list(colors)

    props = {"color": "w", "markeredgewidth": 0, "lw": 0, "markersize": markersize}
    props.update(line2d_kwargs)

    return [
        Line2D([0], [0], marker=marker, markerfacecolor=c, label=str(n), **props)
        for n, c in zip(names, seq)
    ]
