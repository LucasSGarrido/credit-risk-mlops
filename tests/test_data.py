# tests/test_data.py
import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    drop_high_missing,
    encode_categoricals,
    impute_numerics,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 0],
            "AMT_CREDIT": [500000.0, 250000.0, 750000.0],
            "AMT_INCOME_TOTAL": [150000.0, 90000.0, 200000.0],
            "AMT_ANNUITY": [25000.0, 15000.0, 35000.0],
            "DAYS_BIRTH": [-14000, -12000, -16000],
            "DAYS_EMPLOYED": [-2000, -1000, -3000],
            "HIGH_MISSING_COL": [np.nan, np.nan, np.nan],
            "CODE_GENDER": ["M", "F", "M"],
        }
    )


def test_drop_high_missing_removes_cols(sample_df):
    result = drop_high_missing(sample_df, threshold=0.6)
    assert "HIGH_MISSING_COL" not in result.columns


def test_drop_high_missing_keeps_good_cols(sample_df):
    result = drop_high_missing(sample_df, threshold=0.6)
    assert "AMT_CREDIT" in result.columns
    assert "TARGET" in result.columns


def test_impute_numerics_no_nans(sample_df):
    df = sample_df.copy()
    df.loc[0, "AMT_CREDIT"] = np.nan
    result = impute_numerics(df)
    assert result["AMT_CREDIT"].isnull().sum() == 0


def test_encode_categoricals_removes_original(sample_df):
    result = encode_categoricals(sample_df)
    assert "CODE_GENDER" not in result.columns


def test_encode_categoricals_creates_dummies(sample_df):
    result = encode_categoricals(sample_df)
    dummy_cols = [c for c in result.columns if c.startswith("CODE_GENDER")]
    assert len(dummy_cols) >= 1


from src.data.features import (
    add_age_years,
    add_annuity_income_ratio,
    add_credit_income_ratio,
    add_credit_term,
    add_employment_years,
    engineer_features,
)


def test_add_credit_income_ratio(sample_df):
    result = add_credit_income_ratio(sample_df)
    assert "CREDIT_INCOME_RATIO" in result.columns
    # 500000 / (150000 + 1)
    assert result["CREDIT_INCOME_RATIO"].iloc[0] == pytest.approx(500000.0 / 150001.0)


def test_add_annuity_income_ratio(sample_df):
    result = add_annuity_income_ratio(sample_df)
    assert "ANNUITY_INCOME_RATIO" in result.columns
    assert result["ANNUITY_INCOME_RATIO"].iloc[0] == pytest.approx(25000.0 / 150001.0)


def test_add_credit_term(sample_df):
    result = add_credit_term(sample_df)
    assert "CREDIT_TERM" in result.columns
    assert result["CREDIT_TERM"].iloc[0] == pytest.approx(500000.0 / 25001.0)


def test_add_age_years(sample_df):
    result = add_age_years(sample_df)
    assert "AGE_YEARS" in result.columns
    expected = 14000 / 365.25
    assert result["AGE_YEARS"].iloc[0] == pytest.approx(expected, rel=1e-3)


def test_add_employment_years_positive(sample_df):
    result = add_employment_years(sample_df)
    assert "EMPLOYMENT_YEARS" in result.columns
    assert result["EMPLOYMENT_YEARS"].iloc[0] >= 0


def test_engineer_features_adds_all_cols(sample_df):
    result = engineer_features(sample_df)
    expected = [
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_TERM",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
    ]
    for col in expected:
        assert col in result.columns, f"Coluna {col} não encontrada"


def test_engineer_features_no_new_nans(sample_df):
    result = engineer_features(sample_df)
    new_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "AGE_YEARS"]
    for col in new_cols:
        assert result[col].isnull().sum() == 0, f"{col} tem NaN"
