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


def test_r_backend_does_not_restore_workspace(tmp_path):
    # a stale .RData in the working directory must not be loaded into the embedded R
    import shutil
    import subprocess
    import sys

    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("needs R")
    try:
        import rpy2  # noqa: F401
    except ImportError:
        pytest.skip("needs rpy2")

    subprocess.run([rscript, "--no-restore", "-e",
                    "stale_obj <- 1; save(stale_obj, file='.RData')"],
                   cwd=tmp_path, check=True, capture_output=True)
    code = ("import cbrmb.rbackend as rb; ro = rb.require_rpy2()[0]; "
            "print(','.join(ro.r('ls(all.names=TRUE)')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True,
                         capture_output=True, text=True)
    assert "stale_obj" not in out.stdout
