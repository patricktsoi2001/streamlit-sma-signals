"""
data_lib.py
===========
Pure data-processing logic for the Streamlit analysis app.

Separated from the UI so the cleansing / descriptive / diagnostic
functions can be unit-tested without importing Streamlit.

Public API:
    load_csv(bytes, sep) -> pd.DataFrame
    infer_problem_columns(df) -> dict
    do_cleanse(df, cfg) -> pd.DataFrame
    numeric_cols(df) -> list[str]
    categorical_cols(df) -> list[str]
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def load_csv(uploaded_bytes: bytes, sep: str) -> pd.DataFrame:
    """Read an uploaded CSV (raw bytes) into a DataFrame."""
    return pd.read_csv(pd.io.common.BytesIO(uploaded_bytes), sep=sep)


def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Best-effort numeric coercion that also strips thousands separators
    (e.g. '1,200' / '1 200') and currency symbols so text-formatted numbers parse."""
    cleaned = s.astype(str).str.replace(r"[,$£€\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def infer_problem_columns(df: pd.DataFrame) -> Dict:
    """Return a dict of data-quality issues to surface before cleansing."""
    issues: Dict = {}
    issues["missing_cells"] = int(df.isna().sum().sum())
    issues["duplicate_rows"] = int(df.duplicated().sum())
    issues["constant_cols"] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    numeric_looking: List[str] = []
    for c in df.select_dtypes(include="object").columns:
        coerced = _to_numeric_series(df[c])
        if coerced.notna().sum() > 0.5 * len(df):
            numeric_looking.append(c)
    issues["numeric_in_object"] = numeric_looking
    return issues


def do_cleanse(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Apply the selected cleansing steps in a deterministic order.

    cfg keys:
        drop_duplicates: bool
        coerce_numeric: bool
        numeric_cols: list[str]
        missing_strategy: "none" | "drop_rows" | "mean" | "median" | "mode" | "ffill" | "constant"
        fill_value: Optional[str]
        remove_outliers: bool
        outlier_method: "iqr" | "zscore"
    """
    out = df.copy()

    if cfg.get("drop_duplicates"):
        out = out.drop_duplicates()

    if cfg.get("coerce_numeric") and cfg.get("numeric_cols"):
        for c in cfg["numeric_cols"]:
            out[c] = _to_numeric_series(out[c])

    strat = cfg.get("missing_strategy", "none")
    if strat == "drop_rows":
        out = out.dropna()
    elif strat != "none":
        num_cols = out.select_dtypes(include=[np.number]).columns
        if strat in ("mean", "median"):
            out[num_cols] = out[num_cols].fillna(out[num_cols].agg(strat))
        elif strat == "mode":
            for c in out.columns:
                mode = out[c].mode(dropna=True)
                if not mode.empty:
                    out[c] = out[c].fillna(mode.iloc[0])
        elif strat == "ffill":
            out = out.ffill().bfill()
        elif strat == "constant":
            out = out.fillna(cfg.get("fill_value"))

    if cfg.get("remove_outliers"):
        method = cfg.get("outlier_method", "iqr")
        for c in out.select_dtypes(include=[np.number]).columns:
            s = out[c].dropna()
            if len(s) < 4:
                continue
            if method == "iqr":
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            else:  # zscore
                z = (s - s.mean()) / s.std(ddof=0)
                lo = s[z >= -3].min()
                hi = s[z <= 3].max()
            out = out[(out[c] >= lo) & (out[c] <= hi)]

    return out


def numeric_cols(df: pd.DataFrame) -> List[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def categorical_cols(df: pd.DataFrame) -> List[str]:
    obj = df.select_dtypes(include="object")
    return [c for c in obj.columns if df[c].nunique(dropna=True) <= 50]
