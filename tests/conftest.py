import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def two_group_df(rng):
    """20 samples x 3 features; 'hit' is strongly shifted between groups, 'null' is not."""
    n = 20
    ref = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
    shift = np.where(ref == "B", 5.0, 0.0)
    df = pd.DataFrame(
        {
            "hit": rng.normal(0, 1, n) + shift,
            "null": rng.normal(0, 1, n),
            "null2": rng.normal(10, 2, n),
        },
        index=[f"s{i}" for i in range(n)],
    )
    return df, ref


@pytest.fixture
def three_group_df(rng):
    n = 30
    ref = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    shift = np.select([ref == "A", ref == "B", ref == "C"], [0.0, 2.0, 6.0])
    df = pd.DataFrame(
        {
            "hit": rng.normal(0, 1, n) + shift,
            "null": rng.normal(0, 1, n),
        },
        index=[f"s{i}" for i in range(n)],
    )
    return df, ref
