import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from cbrmb.plotting import plot_read_depth


def test_plot_read_depth_creates_figure(rng):
    depths = rng.integers(0, 20000, size=50)
    fig, ax = plot_read_depth(depths, cutoff=5000)
    assert fig is not None
    assert ax.get_xlabel() == "Reads detected"
    assert ax.get_title() == "n=50"


def test_plot_read_depth_reuses_axes(rng):
    depths = rng.integers(0, 20000, size=50)
    fig, ax = mpl.pyplot.subplots()
    out_fig, out_ax = plot_read_depth(depths, ax=ax, inset=False)
    assert out_fig is None
    assert out_ax is ax
    assert len(fig.axes) == 1  # no inset was added
