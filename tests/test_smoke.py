"""Smoke tests:  python -m pytest   (prediction tests need results/koh_models.joblib from 03_final_models.py)"""
import os, sys, subprocess
import numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import load_raw, metrics

HAVE_MODELS = os.path.exists(os.path.join(ROOT, "results", "koh_models.joblib"))
needs_models = pytest.mark.skipif(not HAVE_MODELS, reason="run 03_final_models.py first")


def test_raw_data_is_full_grid():
    df = load_raw()
    assert len(df) == 245                                   # 7 IR x 5 temperatures x 7 times
    assert df.groupby(["IR", "Act_Temp", "Act_Time"]).size().max() == 1
    assert df.Sur_Area.notna().all()
    assert df.Yield.notna().sum() == 40                     # 35 at IR 4 + one each at IR 1, 2, 3, 5, 6


def test_metrics_perfect_prediction():
    m = metrics([1, 2, 3], [1, 2, 3])
    assert m["R2"] == 1 and m["RMSE"] == 0


@needs_models
def test_predict_ir4_has_yield_and_ordered_interval():
    from predict_api import predict
    r = predict(4, 950, 52)
    assert r["sa_lo"] < r["surface_area"] < r["sa_hi"]
    assert r["yield_pct"] is not None and not r["yield_is_estimate"]


@needs_models
@pytest.mark.parametrize("ir", [0, 0.5, 7])
def test_predict_no_yield_outside_ir_1_to_6(ir):
    from predict_api import predict
    r = predict(ir, 950, 52)
    assert r["yield_pct"] is None
    assert any(f"IR={ir:g}" in w for w in r["warnings"])


@needs_models
def test_cli_works_from_another_directory(tmp_path):
    # IR 3 goes through the yield-estimate branch, which reads the raw data file
    out = subprocess.run([sys.executable, os.path.join(ROOT, "predict_api.py"), "--ir", "3", "--temp", "1000", "--time", "60"],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Surface area" in out.stdout and "Yield" in out.stdout
