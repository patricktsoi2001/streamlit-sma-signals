"""
Streamlit Data Analysis App
===========================
Upload a CSV, then run data cleansing, descriptive analysis and
diagnostic analysis with interactive visualizations.

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_lib import (
    load_csv,
    infer_problem_columns,
    do_cleanse,
    numeric_cols,
    categorical_cols,
)

st.set_page_config(
    page_title="CSV Data Analysis Studio",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📊 Data Analysis Studio")
st.sidebar.caption("Upload a CSV to begin.")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
sep_label = st.sidebar.selectbox("Delimiter", ["Comma (,)", "Semicolon (;)", "Tab (\\t)"], index=0)
SEP_MAP = {"Comma (,)": ",", "Semicolon (;)": ";", "Tab (\\t)": "\t"}

if uploaded is not None:
    try:
        raw = load_csv(uploaded.getvalue(), SEP_MAP[sep_label])
        st.session_state.raw_df = raw
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Could not read CSV: {exc}")
        st.session_state.raw_df = None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.title("📊 CSV Data Analysis Studio")
st.markdown(
    "Upload a CSV file, cleanse it, then explore **descriptive** and **diagnostic** "
    "analysis with interactive charts."
)

if st.session_state.raw_df is None:
    st.info("👈 Use the sidebar to upload a CSV file to get started.")
    st.stop()

raw_df = st.session_state.raw_df

# Tabs for workflow
tab_raw, tab_cleanse, tab_descriptive, tab_diagnostic = st.tabs(
    ["1 · Raw Data", "2 · Cleansing", "3 · Descriptive", "4 · Diagnostic"]
)

# ---- Tab 1: Raw Data -------------------------------------------------------
with tab_raw:
    st.subheader("Raw dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", raw_df.shape[0])
    c2.metric("Columns", raw_df.shape[1])
    c3.metric("Missing cells", int(raw_df.isna().sum().sum()))
    c4.metric("Duplicate rows", int(raw_df.duplicated().sum()))

    issues = infer_problem_columns(raw_df)
    if issues["missing_cells"] or issues["duplicate_rows"] or issues["numeric_in_object"]:
        st.warning("**Data quality notes**")
        bullets = []
        if issues["missing_cells"]:
            bullets.append(f"- {issues['missing_cells']} missing cells across the dataset.")
        if issues["duplicate_rows"]:
            bullets.append(f"- {issues['duplicate_rows']} fully duplicated rows.")
        if issues["numeric_in_object"]:
            bullets.append(
                f"- Columns stored as text but look numeric: `{', '.join(issues['numeric_in_object'])}`."
            )
        if issues["constant_cols"]:
            bullets.append(f"- Constant/low-variance columns: `{', '.join(issues['constant_cols'])}`.")
        st.markdown("\n".join(bullets))

    with st.expander("Preview first 50 rows"):
        st.dataframe(raw_df.head(50), use_container_width=True)
    with st.expander("Column types & missing counts"):
        dt = pd.DataFrame({
            "dtype": raw_df.dtypes.astype(str),
            "non_null": raw_df.notna().sum(),
            "missing": raw_df.isna().sum(),
            "n_unique": raw_df.nunique(dropna=True),
        })
        st.dataframe(dt, use_container_width=True)

# ---- Tab 2: Cleansing ------------------------------------------------------
with tab_cleanse:
    st.subheader("Configure cleansing pipeline")
    cfg = {}
    cfg["drop_duplicates"] = st.checkbox("Drop duplicate rows", value=True)
    cfg["coerce_numeric"] = st.checkbox("Convert text columns that look numeric", value=True)
    cfg["numeric_cols"] = (
        issues["numeric_in_object"] if cfg["coerce_numeric"] else []
    )
    cfg["missing_strategy"] = st.selectbox(
        "Missing value strategy",
        ["none", "drop_rows", "mean", "median", "mode", "ffill", "constant"],
        index=3,
        help="mean/median apply to numeric columns; mode applies to all.",
    )
    if cfg["missing_strategy"] == "constant":
        cfg["fill_value"] = st.text_input("Fill value", value="0")
    else:
        cfg["fill_value"] = None
    cfg["remove_outliers"] = st.checkbox("Remove outliers from numeric columns", value=False)
    if cfg["remove_outliers"]:
        cfg["outlier_method"] = st.radio("Outlier method", ["iqr", "zscore"], horizontal=True)
    else:
        cfg["outlier_method"] = "iqr"

    if st.button("Apply cleansing", type="primary"):
        st.session_state.clean_df = do_cleanse(raw_df, cfg)
        st.success("Cleansing applied.")

    if st.session_state.clean_df is not None:
        clean_df = st.session_state.clean_df
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows after", clean_df.shape[0], delta=clean_df.shape[0] - raw_df.shape[0])
        c2.metric("Columns after", clean_df.shape[1])
        c3.metric("Missing cells after", int(clean_df.isna().sum().sum()))

        with st.expander("Cleaned data preview"):
            st.dataframe(clean_df.head(50), use_container_width=True)

        # Allow download of cleaned data
        buf = io.StringIO()
        clean_df.to_csv(buf, index=False)
        st.download_button(
            "Download cleaned CSV",
            data=buf.getvalue(),
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Apply cleansing** to generate the cleaned dataset used by the analysis tabs.")

# Shared: analysis runs on cleaned data if available, else raw
analysis_df = st.session_state.clean_df if st.session_state.clean_df is not None else raw_df
analysis_label = "cleaned" if st.session_state.clean_df is not None else "raw"
if st.session_state.clean_df is None:
    st.caption("⚠️ Showing analysis on **raw** data — run cleansing to analyze the cleaned version.")

# ---- Tab 3: Descriptive ----------------------------------------------------
with tab_descriptive:
    st.subheader(f"Descriptive analysis ({analysis_label} data)")
    num_cols = numeric_cols(analysis_df)
    cat_cols = categorical_cols(analysis_df)

    if num_cols:
        st.markdown("**Numeric summary statistics**")
        desc = analysis_df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
        desc["missing"] = analysis_df[num_cols].isna().sum()
        desc["skew"] = analysis_df[num_cols].skew()
        st.dataframe(desc, use_container_width=True)

        st.markdown("**Distributions**")
        dist_col = st.selectbox("Select a numeric column to visualize", num_cols, key="dist_col")
        b1, b2 = st.columns(2)
        with b1:
            fig = px.histogram(analysis_df, x=dist_col, nbins=30, title=f"Histogram of {dist_col}")
            st.plotly_chart(fig, use_container_width=True)
        with b2:
            fig = px.box(analysis_df, y=dist_col, title=f"Box plot of {dist_col}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Distribution grid (all numeric columns)**")
        for col in num_cols:
            fig = px.histogram(analysis_df, x=col, nbins=20, title=col)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns to summarize.")

    if cat_cols:
        st.divider()
        st.markdown("**Categorical value counts**")
        cat_sel = st.selectbox("Select a categorical column", cat_cols, key="cat_col")
        vc = analysis_df[cat_sel].value_counts().reset_index()
        vc.columns = [cat_sel, "count"]
        fig = px.bar(vc, x=cat_sel, y="count", title=f"Value counts: {cat_sel}")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Full value counts table"):
            st.dataframe(vc, use_container_width=True)
    else:
        st.info("No low-cardinality categorical columns to summarize.")

# ---- Tab 4: Diagnostic -----------------------------------------------------
with tab_diagnostic:
    st.subheader(f"Diagnostic analysis ({analysis_label} data)")
    num_cols = numeric_cols(analysis_df)

    if len(num_cols) >= 2:
        st.markdown("**Correlation matrix**")
        corr = analysis_df[num_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Pearson correlation heatmap",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Strongest correlations**")
        corr_pairs = (
            corr.where(~np.eye(len(corr), dtype=bool))
            .abs()
            .unstack()
            .dropna()
            .sort_values(ascending=False)
        )
        top = corr_pairs.head(10)
        rows = []
        for (a, b), v in top.items():
            rows.append({"var_a": a, "var_b": b, "abs_corr": round(v, 3),
                         "actual_corr": round(corr.loc[a, b], 3)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("**Scatter relationship**")
        s1, s2 = st.columns(2)
        with s1:
            x_col = st.selectbox("X variable", num_cols, key="x_col")
        with s2:
            y_col = st.selectbox("Y variable", [c for c in num_cols if c != x_col], key="y_col")
        color_col = None
        if cat_cols:
            color_col = st.selectbox("Color by (optional)", [None] + cat_cols, key="color_col")
        fig = px.scatter(
            analysis_df, x=x_col, y=y_col, color=color_col,
            title=f"{x_col} vs {y_col}", opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two numeric columns for correlation/scatter diagnostics.")

    # Group-by diagnostic
    if cat_cols and num_cols:
        st.divider()
        st.markdown("**Group comparison (mean of numeric metric by category)**")
        g1, g2 = st.columns(2)
        with g1:
            group_col = st.selectbox("Group by", cat_cols, key="group_col")
        with g2:
            metric_col = st.selectbox("Metric (mean)", num_cols, key="metric_col")
        grouped = analysis_df.groupby(group_col, observed=True)[metric_col].mean().reset_index()
        fig = px.bar(grouped, x=group_col, y=metric_col,
                     title=f"Mean {metric_col} by {group_col}")
        st.plotly_chart(fig, use_container_width=True)

    # Normality / diagnostic stats table
    if num_cols:
        st.divider()
        st.markdown("**Diagnostic statistics**")
        diag_rows = []
        for col in num_cols:
            s = analysis_df[col].dropna()
            diag_rows.append({
                "column": col,
                "mean": round(s.mean(), 3),
                "std": round(s.std(), 3),
                "skew": round(s.skew(), 3),
                "kurtosis": round(s.kurt(), 3),
                "min": round(s.min(), 3),
                "max": round(s.max(), 3),
            })
        st.dataframe(pd.DataFrame(diag_rows), use_container_width=True)
        st.caption(
            "Interpretation: |skew| > 1 suggests strong asymmetry; high kurtosis "
            "indicates heavy tails. Use these to decide on transforms (log, z-score)."
        )

st.sidebar.divider()
st.sidebar.caption(
    "Analysis runs on cleaned data when available, otherwise raw data."
)
