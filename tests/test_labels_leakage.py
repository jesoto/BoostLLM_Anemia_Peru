import pandas as pd
import pytest

from boostllm_anemia.data import assert_no_leakage
from boostllm_anemia.labels import recompute_anemia, altitude_adjustment_who2024
from boostllm_anemia.evaluate import attribution_discrepancy


def test_leakage_guard():
    with pytest.raises(ValueError):
        assert_no_leakage(["HC1", "HC53"])
    assert_no_leakage(["HC1", "HV040"])


def test_recompute_sea_level():
    df = pd.DataFrame({"HC1": [10, 10, 30], "HV040": [100, 100, 100], "HC53": [10.0, 11.0, 10.9]})
    assert recompute_anemia(df).tolist() == [1, 0, 1]  # cortes 10.5 (6-23m) y 11.0 (24-59m)


def test_ads_identical_rankings():
    assert attribution_discrepancy(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_who2024_table4():
    alt = pd.Series([100, 500, 1250, 3250, 4750])
    assert altitude_adjustment_who2024(alt).round(1).tolist() == [0.0, 0.4, 0.8, 2.1, 3.3]


def test_who2024_equation_close_to_table():
    alt = pd.Series([750, 2750, 4750])
    d = altitude_adjustment_who2024(alt) - altitude_adjustment_who2024(alt, use_equation=True)
    assert d.abs().max() < 0.1
