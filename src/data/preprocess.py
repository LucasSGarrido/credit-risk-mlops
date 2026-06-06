# src/data/preprocess.py
"""
Limpeza e preparação dos dados brutos do Home Credit Default Risk.
Entrada: CSV bruto (application_train.csv)
Saída: DataFrame limpo pronto para feature engineering
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

CATEGORICAL_COLS = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
]


def load_raw(filepath: Path) -> pd.DataFrame:
    """Carrega o CSV bruto e retorna DataFrame."""
    return pd.read_csv(filepath)


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Remove colunas com proporção de NaN acima do threshold."""
    missing_ratio = df.isnull().mean()
    cols_to_keep = missing_ratio[missing_ratio < threshold].index.tolist()
    return df[cols_to_keep]


def impute_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa NaN em colunas numéricas com a mediana da coluna."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding em todas as colunas object do DataFrame."""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)


def preprocess(raw_path: Path = DATA_RAW / "application_train.csv") -> pd.DataFrame:
    """Pipeline completo: load → drop_high_missing → impute → encode."""
    df = load_raw(raw_path)
    df = drop_high_missing(df)
    df = impute_numerics(df)
    df = encode_categoricals(df)
    return df


def save_processed(df: pd.DataFrame, name: str = "train") -> Path:
    """Salva DataFrame como parquet em data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    df = preprocess()
    path = save_processed(df)
    print(f"Salvo em {path} — shape: {df.shape}")
