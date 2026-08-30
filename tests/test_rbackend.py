import numpy as np
import pandas as pd
import pytest

from cbrmb.rbackend._bridge import square_array


def test_square_array_from_dataframe_and_validation():
    df = pd.DataFrame(np.zeros((4, 4)), index=list("abcd"), columns=list("abcd"))
    arr = square_array(df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (4, 4)

    with pytest.raises(ValueError, match="square"):
        square_array(np.zeros((3, 5)))


def test_rbackend_submodules_import_without_r():
    # importing cbrmb and the rbackend package must not require rpy2
    import cbrmb.rbackend as rb

    assert hasattr(rb, "require_rpy2")
    assert hasattr(rb, "r_package")
