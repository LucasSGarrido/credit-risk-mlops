# src/data/features.py
"""
Feature engineering: derivar novas features a partir das colunas brutas.
Todas as funções recebem e retornam DataFrames (funções puras, testáveis).
"""

import pandas as pd


def add_credit_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Razão entre valor do crédito e renda anual — indica nível de endividamento."""
    if "AMT_CREDIT" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df = df.copy()
        df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    return df


def add_annuity_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Razão entre parcela mensal e renda — indica comprometimento de renda."""
    if "AMT_ANNUITY" in df.columns and "AMT_INCOME_TOTAL" in df.columns:
        df = df.copy()
        df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    return df


def add_credit_term(df: pd.DataFrame) -> pd.DataFrame:
    """Prazo implícito do crédito em meses (crédito / parcela)."""
    if "AMT_CREDIT" in df.columns and "AMT_ANNUITY" in df.columns:
        df = df.copy()
        df["CREDIT_TERM"] = df["AMT_CREDIT"] / (df["AMT_ANNUITY"] + 1)
    return df


def add_age_years(df: pd.DataFrame) -> pd.DataFrame:
    """Converte DAYS_BIRTH (negativo) para anos de idade."""
    if "DAYS_BIRTH" in df.columns:
        df = df.copy()
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25
    return df


def add_employment_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte DAYS_EMPLOYED para anos de emprego.
    Nota: valor positivo grande (365243) indica desempregado — clip para 0.
    """
    if "DAYS_EMPLOYED" in df.columns:
        df = df.copy()
        df["EMPLOYMENT_YEARS"] = df["DAYS_EMPLOYED"].clip(upper=0).abs() / 365.25
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as transformações de feature engineering em sequência."""
    df = add_credit_income_ratio(df)
    df = add_annuity_income_ratio(df)
    df = add_credit_term(df)
    df = add_age_years(df)
    df = add_employment_years(df)
    return df
