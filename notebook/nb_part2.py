# %% [markdown]
# ## 3. Exploratory Data Analysis (EDA)

# %%
# 3.1 Target distribution --------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
sns.histplot(raw.Sales, bins=60, ax=ax[0], color="#4C72B0"); ax[0].set_title("Order-line Sales"); ax[0].set_yscale("log")
sns.histplot(np.log10(daily[daily > 0]), bins=40, ax=ax[1], color="#55A868"); ax[1].set_title("Daily Sales – log10 (days with orders)")
ax[2].bar(["No orders", "With orders"], [(daily == 0).sum(), (daily > 0).sum()], color=["#C44E52", "#55A868"])
ax[2].set_title("Days with / without orders")
for i, v in enumerate([(daily == 0).sum(), (daily > 0).sum()]): ax[2].text(i, v + 10, str(v), ha="center")
plt.tight_layout(); savefig("02_target_distribution"); plt.show()

# %% [markdown]
# **Observation.** Sales are **heavily right-skewed** (many small lines, a few very large ones) and the daily series is
# **zero-inflated** (≈60 % of days are 0). On the log scale the non-zero days look roughly bell-shaped.
# Consequence: error metrics such as MAPE (divide by actual) are impossible (division by zero), and a model that predicts
# the *average* day is already a strong benchmark – so we will always compare models against simple baselines.

# %%
# 3.2 Sales over time -----------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
ax[0].plot(daily.index, daily.values, lw=.6, color="#9aa5b1", label="daily")
ax[0].plot(daily.rolling(30).mean(), color="#C44E52", lw=2, label="30-day average")
ax[0].plot(daily.rolling(90).mean(), color="#1f3b73", lw=2, label="90-day average"); ax[0].legend(); ax[0].set_title("Daily sales")
monthly = daily.resample("MS").sum()
ax[1].bar(monthly.index, monthly.values, width=25, color="#4C72B0"); ax[1].set_title("Monthly total sales")
plt.tight_layout(); savefig("03_sales_over_time"); plt.show()
yearly = daily.groupby(daily.index.year).sum()
display(yearly.to_frame("total sales"))

# %% [markdown]
# **Observation.** (1) **Strong growth** from 2011 (business ramp-up) to 2013–2014 – the series is *not stationary*, it has a trend.
# (2) Sales arrive in **bursts**: weeks with many orders, followed by empty weeks. (3) Some months stand out (see seasonality below).
# 2011 and 2014 are *incomplete years* (history starts on 21-Jan-2011 and ends on 29-Sep-2014).

# %%
# 3.3 Seasonality: month and weekday ---------------------------------------------------------------------------------
tmp = daily.to_frame("sales")
tmp["month"] = tmp.index.month; tmp["dow"] = tmp.index.dayofweek; tmp["year"] = tmp.index.year
pivot = tmp.pivot_table(index="month", columns="year", values="sales", aggfunc="sum")
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
for y in pivot.columns: ax[0].plot(pivot.index, pivot[y], marker="o", label=str(y))
ax[0].set_title("Monthly sales by year"); ax[0].set_xlabel("month"); ax[0].legend()
dow = tmp.groupby("dow").sales.agg(["mean", lambda s: (s > 0).mean()]); dow.columns = ["avg_sales", "share_days_with_orders"]
ax[1].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.avg_sales, color="#4C72B0"); ax[1].set_title("Average sales per weekday")
ax[2].bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.share_days_with_orders * 100, color="#DD8452")
ax[2].set_title("% of days with orders, per weekday")
plt.tight_layout(); savefig("04_seasonality"); plt.show()
full_years = pivot[[2012, 2013]].mean(axis=1)
print("Strongest months (avg of 2012 & 2013):", full_years.nlargest(3).round(0).to_dict())
print("Weakest months (avg of 2012 & 2013)  :", full_years.nsmallest(3).round(0).to_dict())
print(f"Weekday spread: best avg = {dow.avg_sales.max():,.0f}, worst avg = {dow.avg_sales.min():,.0f}")
RESULTS["strong_months"] = {int(k): float(v) for k, v in full_years.nlargest(3).items()}

# %% [markdown]
# **Observation.** There is a clear **yearly pattern**: averaged over the two complete years 2012 and 2013, **December, September and July are the
# strongest months and November, May and February the weakest** (printed above). The **weekday effect is weak**: orders occur on all seven days and the differences between
# weekdays are small compared with the day-to-day noise. So month-of-year/trend features matter more than day-of-week features.

# %%
# 3.4 Box plots: spread by weekday & month ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
nz = tmp[tmp.sales > 0]
sns.boxplot(data=nz, x="dow", y="sales", ax=ax[0], color="#8ab4d8"); ax[0].set_yscale("log"); ax[0].set_title("Sales on days with orders – by weekday")
ax[0].set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
sns.boxplot(data=nz, x="month", y="sales", ax=ax[1], color="#a8d8b0"); ax[1].set_yscale("log"); ax[1].set_title("Sales on days with orders – by month")
plt.tight_layout(); savefig("05_boxplots_calendar"); plt.show()

# %% [markdown]
# **Observation.** The medians are very similar across weekdays and months (log scale!), and every group contains high outliers.
# Differences between months come mostly from **how many days have orders** and a few huge days, not from a much higher "normal" day.

# %%
# 3.5 Categorical analysis (count plots + sales by category/country/product) --------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(15, 9))
order = raw.CategoryName.value_counts().index
sns.countplot(data=raw, y="CategoryName", order=order, ax=ax[0, 0], color="#4C72B0"); ax[0, 0].set_title("Count plot – order lines per category")
cat_sales = raw.groupby("CategoryName").Sales.sum().sort_values()
ax[0, 1].barh(cat_sales.index, cat_sales.values, color="#55A868"); ax[0, 1].set_title("Total sales per category")
top_c = raw.groupby("Country").Sales.sum().nlargest(10).sort_values()
ax[1, 0].barh(top_c.index, top_c.values, color="#DD8452"); ax[1, 0].set_title("Top-10 countries by sales")
top_p = raw.groupby("ProductName").Sales.sum().nlargest(10).sort_values()
ax[1, 1].barh(top_p.index, top_p.values, color="#8172B3"); ax[1, 1].set_title("Top-10 products by sales")
plt.tight_layout(); savefig("06_categorical_analysis"); plt.show()
share_top3_c = raw.groupby("Country").Sales.sum().nlargest(3).sum() / raw.Sales.sum()
print(f"Top-3 countries = {share_top3_c:.0%} of all sales | top-10 products = {top_p.sum()/raw.Sales.sum():.0%} of all sales")

# %% [markdown]
# **Observation.** Sales are fairly spread over the 8 product categories, but **a few countries and products generate a large share**
# (printed above). For the *total-sales* forecast this is background information; it would matter if we forecast per country/category.

# %%
# 3.6 Feature vs target ----------------------------------------------------------------------------------------------
agg = raw.groupby("OrderDate").agg(sales=("Sales", "sum"), quantity=("Quantity", "sum"), discount=("Discount", "sum"),
                                   cos=("COS", "sum"), gp=("GP", "sum"), freight=("Freight", "sum"),
                                   n_orders=("Order", "nunique"), n_lines=("Order", "size"))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(raw.Quantity, raw.Sales, s=8, alpha=.4); ax[0].set_xlabel("Quantity"); ax[0].set_ylabel("Sales"); ax[0].set_title("Line level: Quantity vs Sales")
ax[1].scatter(agg.n_orders, agg.sales, s=10, alpha=.5, color="#55A868"); ax[1].set_xlabel("orders on the day"); ax[1].set_title("Daily: #orders vs Sales")
ax[2].scatter(agg.gp, agg.sales, s=10, alpha=.5, color="#C44E52"); ax[2].set_xlabel("gross profit of the day"); ax[2].set_title("Daily: GP vs Sales (leakage!)")
plt.tight_layout(); savefig("07_feature_vs_target"); plt.show()

# %% [markdown]
# **Observation.** Same-day quantity, number of orders and profit are strongly related to same-day sales – but that relationship is
# *circular* (they are produced by the same orders). The GP-vs-Sales panel is almost a straight line, which is exactly why
# those columns are **leakage** and are not allowed as model inputs for a future date.

# %% [markdown]
# ## 4. Correlation Analysis

# %%
corr = agg.corr()
plt.figure(figsize=(8, 6.5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation – daily aggregated numeric columns"); plt.tight_layout(); savefig("08_corr_raw"); plt.show()
pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
print("Highly correlated pairs (|r| > 0.90):"); display(pairs[pairs.abs() > .9].sort_values(ascending=False).to_frame("r"))
print("Weak / no correlation with Sales (|r| < 0.20):", corr["sales"].drop("sales")[corr["sales"].drop("sales").abs() < .2].round(2).to_dict())
print("Negative correlation with Sales:", corr["sales"].drop("sales")[corr["sales"].drop("sales") < 0].round(2).to_dict())

# %% [markdown]
# **Observation.** `cos` (r = 1.00) and `gp` (r = 0.95) are almost perfectly correlated with `sales` because `sales = cos + gp`, and `cos` and `gp`
# are correlated with each other (0.92) → **multicollinearity *and* leakage**. `freight` and `n_lines` are also highly correlated (0.92): each shipment
# carries freight, so more order lines mean more freight. All other same-day columns (quantity, discount, number of orders…) are also positively
# correlated with sales – busy days are busy in *every* column. There are **no negative and no near-zero correlations**, which is itself a sign that these
# columns all measure the same thing (the size of the day). Since none of them is known for a future date, they are removed from the model inputs.

# %% [markdown]
# ## 5. Feature Engineering
# Because the future values of `quantity`, `discount`… are unknown, we create **features that are knowable for any future date**:
#
# | Group | Features | Why |
# |---|---|---|
# | Trend | `t` (days since first sale) | captures long-term growth |
# | Calendar | day of week, weekend flag, day of month, week of year, month, quarter, day of year, month-start / month-end / quarter-end, December flag | business cycles (pay-day, month-end ordering, year-end) |
# | Cyclical encoding | sin / cos of day-of-year (1st & 2nd harmonic), weekday, month | tells the model that Dec 31 is next to Jan 1 |
# | Lag features | sales 1, 7, 14, 28 days ago | recent history |
# | Rolling features | mean of last 7 / 14 / 28 / 90 days, std 28, share of days with orders (28), days since last sale | current sales *level* and volatility |
#
# Lag/rolling features are shifted by one day, so **a day never sees its own sales** (no leakage).
# Categorical variables need no one-hot encoding because every calendar feature is numeric; scale-sensitive models receive a
# `StandardScaler` *inside their pipeline* (fitted only on training data).
#
# We build **two feature sets** and compare them:
# * **Set A – calendar only** : can predict all 90 days in one shot ("direct" forecast).
# * **Set B – calendar + lags** : needs *recursive* forecasting (predict day 1, feed it back, predict day 2 …), so errors can accumulate.

# %%
print(inspect.getsource(fc.lag_features))

# %%
XA = fc.build_frame(daily, use_lags=False); XB = fc.build_frame(daily, use_lags=True)
print("Set A:", XA.shape, "| Set B:", XB.shape, "| NaNs in B:", int(XB.isna().sum().sum()))
XB.tail(3).T.round(2)

# %%
# correlation of every engineered feature with the target (Pearson + Spearman)
yB = daily.loc[XB.index]
cp = XB.corrwith(yB); cs = XB.corrwith(yB, method="spearman")
cc = pd.DataFrame({"Pearson": cp, "Spearman": cs}).sort_values("Pearson")
cc.plot.barh(figsize=(9, 9), width=.8); plt.title("Correlation of engineered features with daily sales"); plt.tight_layout()
savefig("09_feature_target_corr"); plt.show()
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(XB.corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": .7}); ax.set_title("Correlation between engineered features")
plt.tight_layout(); savefig("10_feature_heatmap"); plt.show()
cf = XB.corr().where(np.triu(np.ones((XB.shape[1],) * 2), 1).astype(bool)).stack()
print(f"{(cf.abs() > .9).sum()} feature pairs have |r| > 0.90, e.g.:"); print(cf[cf.abs() > .9].abs().sort_values(ascending=False).head(8).round(2).to_string())

# %% [markdown]
# **Observation.** (1) *Every* single feature has a **weak** correlation with the noisy daily target (|r| well below 0.3) – the
# strongest ones are the trend `t` and the rolling means (they describe the sales *level*). That is expected for a spiky daily series.
# (2) Many features are **highly correlated with each other** (month ↔ quarter ↔ day-of-year ↔ week-of-year ↔ sin/cos: eight pairs with |r| > 0.9) → multicollinearity. It does not hurt tree models; linear models need **regularisation** (Ridge/Lasso…) –
# and the feature-selection step below removes redundant columns.

# %% [markdown]
# ## 6. Feature Selection
# We rank the features with three independent methods (**correlation**, **mutual information**, **Random-Forest importance**),
# combine them into an *average rank*, and later (Section 10) test whether keeping only the top-*k* features improves the models
# ("before vs after feature selection"). Selection uses **training data only** (everything before the final 90-day test window)
# to avoid leakage.

# %%
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
TEST_START = daily.index[-HORIZON]
train_only = daily[daily.index < TEST_START]; train_cap = train_only.clip(upper=train_only.quantile(.99))
XBt = fc.build_frame(train_cap, use_lags=True); yBt = train_cap.loc[XBt.index]
mi = pd.Series(mutual_info_regression(XBt, yBt, random_state=SEED), index=XBt.columns)
rf = RandomForestRegressor(300, min_samples_leaf=10, random_state=SEED, n_jobs=1).fit(XBt, yBt)
rfi = pd.Series(rf.feature_importances_, index=XBt.columns)
cor = XBt.corrwith(yBt).abs()
sel = pd.DataFrame({"abs_corr": cor, "mutual_info": mi, "rf_importance": rfi})
sel["avg_rank"] = sel.rank(ascending=False).mean(axis=1)
sel = sel.sort_values("avg_rank"); RESULTS["feature_ranking"] = sel.round(4).reset_index().rename(columns={"index": "feature"}).to_dict("records")
display(sel.round(4).head(15))
fig, ax = plt.subplots(figsize=(8, 8)); sel["avg_rank"].iloc[::-1].plot.barh(ax=ax, color="#4C72B0")
ax.set_title("Feature ranking (lower average rank = more useful)"); plt.tight_layout(); savefig("11_feature_selection"); plt.show()
RANKED = sel.index.tolist()
