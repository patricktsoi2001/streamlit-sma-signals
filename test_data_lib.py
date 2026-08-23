"""Unit tests for data_lib.py — run with: python test_data_lib.py"""

import sys
import pandas as pd
import numpy as np

from data_lib import (
    load_csv,
    infer_problem_columns,
    do_cleanse,
    numeric_cols,
    categorical_cols,
)

CSV = b"""id,price,qty,category,notes
1,"1,200",3,A,ok
2,"1,500",5,B,ok
3,"2,000",,A,ok
1,"1,200",3,A,ok
4,bad,7,C,ok
5,"3,000",9,B,ok
6,"900",2,A,ok
7,"50000",1,C,ok
"""

failures = []


def check(name, cond):
    if cond:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}")
        failures.append(name)


# --- load_csv ---
df = load_csv(CSV, sep=",")
check("load_csv rows", df.shape[0] == 8)
check("load_csv cols", df.shape[1] == 5)

# --- infer_problem_columns ---
issues = infer_problem_columns(df)
check("missing cells detected", issues["missing_cells"] >= 1)
check("duplicate rows detected", issues["duplicate_rows"] == 1)
check("numeric_in_object detects price", "price" in issues["numeric_in_object"])

# --- do_cleanse: coerce + dedup + fill median ---
cfg = {
    "drop_duplicates": True,
    "coerce_numeric": True,
    "numeric_cols": issues["numeric_in_object"],
    "missing_strategy": "median",
    "fill_value": None,
    "remove_outliers": False,
    "outlier_method": "iqr",
}
clean = do_cleanse(df, cfg)
check("cleansing drops duplicate", clean.shape[0] == 7)
check("price coerced to numeric", pd.api.types.is_numeric_dtype(clean["price"]))
check("missing qty filled (no NaN)", clean["qty"].isna().sum() == 0)
check("bad coerced to NaN then filled", clean["price"].isna().sum() == 0)

# --- outlier removal (iqr) should drop the 50000 row ---
cfg2 = dict(cfg, remove_outliers=True, outlier_method="iqr")
clean2 = do_cleanse(df, cfg2)
check("iqr removes outlier row", clean2.shape[0] == 6)
check("iqr keeps normal rows", clean2["price"].max() < 10000)

# --- numeric_cols / categorical_cols ---
check("numeric_cols picks price/qty", set(numeric_cols(clean)) >= {"price", "qty"})
check("categorical_cols picks category", "category" in categorical_cols(clean))
# High-cardinality cutoff (>50 unique) should be excluded
big = pd.DataFrame({"group": [f"g{i % 3}" for i in range(60)], "uuid": [f"u{i}" for i in range(60)]})
check(">50 unique column excluded", "uuid" not in categorical_cols(big))
check("low-cardinality group included", "group" in categorical_cols(big))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
