# %% [markdown]
# # Sales Forecasting – 90-Day Daily Forecast (End-to-End ML Project)
#
# **Goal:** predict the company's **total daily sales for the next 90 days (about 3 months)** using regression models.
#
# **Pipeline:** Data understanding → Cleaning → EDA → Correlation → Feature engineering → Feature selection →
# Time-based split → Compare *all* regression models → Evaluation → Hyper-parameter tuning → Final model selection →
# Save model → FastAPI (Docker / Render) → Streamlit UI.
#
# **Important idea – this is a *time-series regression* problem.** Rows are ordered in time, so we must never shuffle the data
# or let the model see the future. Every split, feature and metric in this notebook respects that rule.

# %%
import sys, os, json, time, warnings, inspect, hashlib, datetime as dt
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import joblib

sys.path.insert(0, "../api")           # shared feature-engineering code (also used by the API)
import forecast_core as fc

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold", "axes.titlesize": 12})
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")

os.makedirs("../docs/figures", exist_ok=True)
os.makedirs("../api/artifacts", exist_ok=True)
def savefig(name):
    plt.savefig(f"../docs/figures/{name}.png", bbox_inches="tight", dpi=130)

DATA = "../data/New_Query_2026_10_22_10_01_37.csv"
HORIZON = 90          # forecast length in days
SEED = 42
RESULTS = {}          # numbers collected here are re-used in the documentation

# %% [markdown]
# ## 1. Data Understanding

# %%
raw = pd.read_csv(DATA)
print(f"Rows: {raw.shape[0]:,}   Columns: {raw.shape[1]}")
raw.head(3).T

# %%
num_cols = raw.select_dtypes(include="number").columns.tolist()
cat_cols = raw.select_dtypes(exclude="number").columns.tolist()
print(f"Numerical columns ({len(num_cols)}): {num_cols}\n")
print(f"Categorical / text columns ({len(cat_cols)}): {cat_cols}")
pd.DataFrame({"dtype": raw.dtypes.astype(str), "n_unique": raw.nunique(), "n_missing": raw.isna().sum()})

# %% [markdown]
# **What the data is.** Each row is one *order line* (one product inside one order) of a clothing wholesaler:
# order id, product, quantity, **Sales**, discount, cost of sales (COS), gross profit (GP), customer, country, employee, shipper, etc.
#
# **Target variable.** `Sales` is the revenue of a line. Because the business question is *"how much will we sell each day?"*,
# the target we model is **total Sales per calendar day** (all order lines of a day added together).
#
# This is a **regression** problem (continuous target), so "class distribution" is replaced by the *target distribution*.

# %%
raw["OrderDate"] = pd.to_datetime(raw["OrderDate"], utc=True).dt.tz_localize(None).dt.normalize()
raw.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(2)

# %%
daily = fc.load_daily_sales(DATA)
print(f"Daily series: {len(daily)} days  ({daily.index.min().date()} → {daily.index.max().date()})")
print(f"Days with at least one order : {(daily>0).sum()}  ({(daily>0).mean():.1%})")
print(f"Days with NO order (sales=0) : {(daily==0).sum()}  ({(daily==0).mean():.1%})")
print(f"Total sales in history       : {daily.sum():,.0f}")
RESULTS["n_rows"] = int(raw.shape[0]); RESULTS["n_cols"] = int(raw.shape[1])
RESULTS["n_days"] = int(len(daily)); RESULTS["zero_day_share"] = float((daily == 0).mean())
RESULTS["data_start"] = str(daily.index.min().date()); RESULTS["data_end"] = str(daily.index.max().date())
daily.describe().to_frame("daily sales")

# %% [markdown]
# **Observation.** The raw file has 2,172 order lines but only 533 distinct order dates spread over 1,348 calendar days.
# So **about 60 % of the days have zero sales** – the daily series is *sparse and spiky*. This is the single most important fact
# for the whole project: a daily forecast can only predict the *expected* sales of a day, not the exact spikes.

# %% [markdown]
# ## 2. Data Cleaning
# Each operation below says **what** is checked and **why**.

# %%
# 2.1 Missing values -------------------------------------------------------------------------------
miss = raw.isna().sum(); miss = miss[miss > 0].to_frame("missing"); miss["%"] = miss["missing"] / len(raw) * 100
display(miss)
print("Missing in OrderDate / Sales:", raw[["OrderDate", "Sales"]].isna().sum().to_dict())

# %% [markdown]
# *Why:* missing values break many models. The only columns with gaps (`ReportsTo`, `Fax`, `Phone`, `PostalCode`) are
# contact/organisation details that have **no influence on total daily sales**, so they are simply not used – no imputation is needed.
# `OrderDate` and `Sales` (the only two columns the forecast needs) are 100 % complete.

# %%
# 2.2 Duplicates -------------------------------------------------------------------------------------
print("Fully duplicated rows            :", raw.duplicated().sum())
print("Duplicated (Order, ProductID) key:", raw.duplicated(["Order", "ProductID"]).sum())

# %% [markdown]
# *Why:* duplicated order lines would count the same revenue twice and inflate daily totals. None were found.

# %%
# 2.3 Data types ---------------------------------------------------------------------------------------
print("OrderDate was stored as text with a UTC time-zone suffix, e.g.:", pd.read_csv(DATA, nrows=1)["OrderDate"][0])
print("Now:", raw["OrderDate"].dtype, "| all timestamps at midnight:", bool((raw["OrderDate"] == raw["OrderDate"].dt.normalize()).all()))
print("PostalCode is float (should be an identifier, not a number):", raw["PostalCode"].dtype)

# %% [markdown]
# *Why:* dates must be real `datetime` objects to build calendar features and resample. The time-zone is removed so that
# 2014-09-29 00:00 UTC is treated as the calendar day 2014-09-29. `PostalCode` is an identifier (not a quantity) and is not used.

# %%
# 2.4 Invalid values & internal consistency -------------------------------------------------------------
checks = {
    "Sales <= 0": (raw.Sales <= 0).sum(), "Quantity <= 0": (raw.Quantity <= 0).sum(),
    "Discount < 0": (raw.Discount < 0).sum(), "COS < 0": (raw.COS < 0).sum(), "GP < 0": (raw.GP < 0).sum(),
    "Freight < 0": (raw.Freight < 0).sum(),
}
print(checks)
print("max |Sales - COS - GP| =", (raw.Sales - raw.COS - raw.GP).abs().max().round(6))

# %% [markdown]
# *Why:* negative or zero revenue/quantity would signal returns or data-entry errors. There are none.
# Notice also that **`Sales = COS + GP`** (to 4-decimal rounding) – the cost and profit columns are just a split of the sales figure
# (this matters for the *leakage* check below).

# %%
# 2.5 Inconsistent categorical values -----------------------------------------------------------------------
print("CategoryName values:", sorted(raw.CategoryName.unique()))
print("Country vs CountryCode unique:", raw.Country.nunique(), raw.CountryCode.nunique())
print("Same name written with different case/spaces:",
      (raw.CategoryName.str.strip().str.lower().nunique() != raw.CategoryName.nunique()))
print("Categories per CategoryDescription (should be 1):", raw.groupby("CategoryName").CategoryDescription.nunique().max())

# %% [markdown]
# *Why:* "Women's wear" and "womens wear " would be counted as two categories. The categories are consistent
# (the only oddity is a typographic accent used instead of an apostrophe in names like *Men´s Clothes*, which is harmless
# and only used for charts).

# %%
# 2.6 Outliers ---------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
sns.boxplot(x=raw.Sales, ax=ax[0], color="#4C72B0"); ax[0].set_title("Order-line Sales"); ax[0].set_xscale("log")
sns.boxplot(x=daily[daily > 0], ax=ax[1], color="#55A868"); ax[1].set_title("Daily Sales (days with orders)"); ax[1].set_xscale("log")
plt.tight_layout(); savefig("01_outliers_boxplots"); plt.show()
q99 = daily.quantile(.99)
print(f"99th percentile of daily sales: {q99:,.0f}   |   max daily sales: {daily.max():,.0f}   |   days above q99: {(daily>q99).sum()}")
daily.nlargest(8).to_frame("sales").assign(weekday=lambda d: d.index.day_name())

# %% [markdown]
# **Decision on outliers.** A handful of days (e.g. 2013-12-29 with ~60k, while a typical busy day is ~2–4k) are real big orders,
# not errors, so they are **kept in the data**. But a model trained with the squared-error loss can be dragged around by the 14 days above the 99th percentile.
# Therefore, **only during training**, the daily target is *winsorised* (capped) at the 99th percentile of the training window.
# Models are always *evaluated on the real, uncapped sales*. A later experiment (Section 12) checks that this choice does not hurt.

# %% [markdown]
# **2.7 Unnecessary columns.** For a forecast of *total daily sales* we only need the date and the sales value.
# Customer contact details (`ContactName`, `Fax`, `Phone`, `PostalCode`), employee details (`EmployeeName`, `Extension`, `HireDate`,
# `YearSalary`, `Office`, `Title`, `ReportsTo`), geography (`Latitude`, `Longitude`) and IDs are not used as model inputs
# (they are personal/descriptive and unknown for future days). Product, category and country are kept **for EDA only**.
#
# **2.8 Leakage detection.** `Quantity`, `Discount`, `COS`, `GP`, `Freight` and the number of orders are *consequences of the same
# day's sales* – and `Sales = COS + GP`. They are unknown for a future date, so using them as inputs would make the model
# look perfect in the notebook and fail in production. **They are excluded from the features.**
# Only information available *before* the day (calendar, and past sales for lag features) is allowed.
