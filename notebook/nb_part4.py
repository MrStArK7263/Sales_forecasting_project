# %% [markdown]
# **Observation (model comparison).**
# 1. Daily sales are dominated by random spikes, so *no* model reaches a meaningful daily R² (all are around 0). The best models beat the best simple benchmark
#    on daily RMSE by only about 1 % – smaller than the fold-to-fold variation – so the models are compared on **several** metrics and always next to the **baselines**.
# 2. **Simple, regularised linear models on calendar features rank first.** Flexible models (Random Forest, boosting, neural network, Gaussian process) fit the
#    noise in the spikes and rank lower; they would need much more data. The lag/rolling features of Set B did not help on average (see the feature-set table):
#    recursive forecasting over 90 days lets small errors feed back into the lag features.
# 3. `RANSAC` failed on every fold (it needs a consensus set, impossible when 60 % of the target is 0) – it is reported as failed and excluded.
# 4. The top-ranked combinations that have a tuning grid are carried to feature selection and tuning below.

# %% [markdown]
# ## 10. Feature selection – before vs after
# For the 3 best-ranked (tunable) combinations we compare **all features** with the **top-k features** from Section 6 (k = 8, 12, 16),
# using the validation folds only.

# %%
from itertools import product
def cv_eval(est, use_lags, cols=None, cap_q=0.99):
    """Mean validation metrics over folds 2-4 for one configuration."""
    res = []
    for k in VAL_FOLDS:
        tr, te = FOLDS[k]; _, p, _, _ = fit_and_forecast(est, tr, HORIZON, use_lags, cols, cap_q); res.append(evaluate(te, p))
    r = pd.DataFrame(res); out = r.mean(); out["Total90_abs"] = r.Total90_err.abs().mean(); return out

GRIDS = {
 "Ridge": {"model__alpha": [.01, .1, 1, 10, 100, 1000, 5000]}, "Lasso": {"model__alpha": [.5, 1, 5, 20, 100, 300]},
 "ElasticNet": {"model__alpha": [5, 20, 100], "model__l1_ratio": [.2, .5, .8]},
 "Bayesian Ridge": {"model__alpha_2": [1e-6, 1e-3], "model__lambda_2": [1e-6, 1e-2, 1]},
 "Huber": {"model__alpha": [1, 10, 100]}, "Poisson GLM": {"model__alpha": [.1, 1, 10, 100]},
 "Tweedie GLM (p=1.5)": {"model__alpha": [.1, 1, 10, 100], "model__power": [1.2, 1.5, 1.8]},
 "KNN": {"model__n_neighbors": [10, 25, 50, 100], "model__weights": ["uniform", "distance"]},
 "SVR (RBF)": {"model__C": [200, 2000, 20000], "model__epsilon": [10, 50, 200]},
 "Kernel Ridge (RBF)": {"model__alpha": [1, 10, 100], "model__gamma": [.005, .02, .05]},
 "Random Forest": {"model__min_samples_leaf": [5, 10, 20, 40], "model__max_features": [.5, 1.0]},
 "Extra Trees": {"model__min_samples_leaf": [5, 10, 20, 40], "model__max_features": [.5, 1.0]},
 "Gradient Boosting": {"model__learning_rate": [.02, .05], "model__n_estimators": [100, 200], "model__max_depth": [2, 3]},
 "HistGradient Boosting": {"model__learning_rate": [.03, .08], "model__max_depth": [2, 3, 4], "model__min_samples_leaf": [20, 50]},
 "XGBoost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [.02, .05], "model__n_estimators": [150, 300]},
 "LightGBM": {"model__num_leaves": [4, 8, 16], "model__learning_rate": [.02, .05], "model__min_child_samples": [20, 50]},
 "CatBoost": {"model__depth": [3, 4, 6], "model__learning_rate": [.03, .08]},
 "Decision Tree": {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [10, 20, 40]},
 "AdaBoost": {"model__learning_rate": [.01, .05, .2], "model__n_estimators": [50, 100]},
 "MLP (neural net)": {"model__alpha": [.1, 1, 10], "model__hidden_layer_sizes": [(16,), (32, 16)]},
}
tunable = cand[cand.model.isin(GRIDS)].drop_duplicates("model").head(3)
print("Candidates carried to feature selection + tuning:"); display(tunable[["model", "feature_set", "RMSE", "WAPE_30d", "Total90_abs", "score"]].round(2))

# %%
fs_rows, CHOSEN = [], {}
for _, r in tunable.iterrows():
    use_lags = r.feature_set.startswith("B"); allcols = fc.feature_names(use_lags); ranked = [c for c in RANKED if c in allcols]
    best_name, best_val = "all features", None
    options = {"all features": allcols, **{f"top {k}": ranked[:k] for k in (8, 12, 16) if k < len(allcols)}}
    res = {}
    for nm, cols in options.items():
        s = cv_eval(ZOO[r.model][0], use_lags, cols); res[nm] = s
        fs_rows.append({"model": r.model, "feature_set": r.feature_set, "features": nm, "n_features": len(cols), "RMSE": s.RMSE,
                        "WAPE_7d": s.WAPE_7d, "WAPE_30d": s.WAPE_30d, "Total90_abs": s.Total90_abs})
    tbl = pd.DataFrame(res).T[["RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs"]]
    pick = tbl.rank().mean(axis=1).idxmin()                           # same average-rank rule
    CHOSEN[r.model] = {"use_lags": use_lags, "cols": options[pick], "choice": pick, "feature_set": r.feature_set}
FS = pd.DataFrame(fs_rows); display(FS.round(2))
print({k: v["choice"] for k, v in CHOSEN.items()})
FS.to_csv("../docs/feature_selection_comparison.csv", index=False)

# %% [markdown]
# **Observation.** The table compares *before* (all features) and *after* (top-k). If a smaller subset has equal or better validation
# error, it is preferred: fewer inputs = simpler, faster and less prone to over-fitting. The chosen subset per model is printed above and is used for tuning.

# %% [markdown]
# ## 11. Hyper-parameter tuning
# Grid search with **time-aware validation** (folds 2-4, never the test fold). `GridSearchCV` with random folds would leak the future,
# so the grid is evaluated with our own rolling-origin loop. For every grid point the same 4-metric average-rank rule is used.

# %%
TUNED, tune_log = {}, []
for name, cfg in CHOSEN.items():
    grid = GRIDS[name]; keys = list(grid); best = None; rows_g = []
    for vals in product(*grid.values()):
        params = dict(zip(keys, vals)); est = clone(ZOO[name][0]).set_params(**params)
        s = cv_eval(est, cfg["use_lags"], cfg["cols"]); rows_g.append({**params, **s[["RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs"]].to_dict()})
    G = pd.DataFrame(rows_g); G["score"] = G[["RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs"]].rank().mean(axis=1)
    b = G.sort_values("score").iloc[0]; bp = {k: (b[k] if not isinstance(b[k], np.generic) else b[k].item()) for k in keys}
    for k in keys:                                       # restore tuples / ints lost by DataFrame conversion
        if isinstance(grid[k][0], tuple): bp[k] = [g for g in grid[k] if str(g) == str(b[k])][0]
        elif isinstance(grid[k][0], int) and not isinstance(grid[k][0], bool): bp[k] = int(bp[k])
    tuned = clone(ZOO[name][0]).set_params(**bp)
    before = cv_eval(ZOO[name][0], cfg["use_lags"], cfg["cols"]); after = cv_eval(tuned, cfg["use_lags"], cfg["cols"])
    TUNED[name] = {"est": tuned, "params": bp, **cfg}
    tune_log.append({"model": name, "feature_set": cfg["feature_set"], "features": cfg["choice"], "best_params": {k.replace("model__", ""): v for k, v in bp.items()},
                     "grid_points": len(G), "RMSE_before": before.RMSE, "RMSE_after": after.RMSE, "WAPE30_before": before.WAPE_30d, "WAPE30_after": after.WAPE_30d,
                     "Total90abs_before": before.Total90_abs, "Total90abs_after": after.Total90_abs})
    print(f"{name:<22} best params: {tune_log[-1]['best_params']}")
TL = pd.DataFrame(tune_log); TL.to_csv("../docs/tuning_results.csv", index=False)
display(TL.drop(columns=["best_params"]).round(2))

# Top-3 combinations that have NO hyper-parameters (e.g. plain Linear Regression) cannot be tuned, but they must not be dropped:
# they join the final comparison exactly as they are.
for _, r in cand.head(3).iterrows():
    if r.model not in GRIDS and r.model not in TUNED:
        ub = r.feature_set.startswith("B")
        TUNED[r.model] = {"est": ZOO[r.model][0], "params": {}, "use_lags": ub, "cols": fc.feature_names(ub), "choice": "all features", "feature_set": r.feature_set}
        tune_log.append({"model": r.model, "feature_set": r.feature_set, "features": "all features", "best_params": {}, "grid_points": 0})
        print(f"{r.model} ({r.feature_set}) has no hyper-parameters – carried to the final comparison un-tuned.")

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); x = np.arange(len(TL)); w = .38
ax[0].bar(x - w / 2, TL.RMSE_before, w, label="before tuning", color="#9aa5b1"); ax[0].bar(x + w / 2, TL.RMSE_after, w, label="after tuning", color="#4C72B0")
ax[0].set_xticks(x); ax[0].set_xticklabels(TL.model, rotation=15); ax[0].set_title("Validation RMSE"); ax[0].legend(); ax[0].set_ylim(TL[["RMSE_before", "RMSE_after"]].min().min() * .95)
ax[1].bar(x - w / 2, TL.Total90abs_before, w, color="#9aa5b1"); ax[1].bar(x + w / 2, TL.Total90abs_after, w, color="#55A868")
ax[1].set_xticks(x); ax[1].set_xticklabels(TL.model, rotation=15); ax[1].set_title("Validation |90-day total error| %")
plt.tight_layout(); savefig("14_tuning_before_after"); plt.show()

# %% [markdown]
# ## 12. Final model selection
# The model with the lowest error is **not automatically** the deployment choice. Candidates are compared on accuracy (validation), stability
# (variation across folds), the untouched **test window**, inference time and complexity. The rule used in code: *lowest average rank over
# RMSE / WAPE-7d / WAPE-30d / |Total-90d error| on the validation folds; ties are resolved in favour of the faster, simpler model.* The test window is only
# used to **confirm** the choice, never to make it.

# %%
def complexity(est):
    m = est.named_steps["model"] if hasattr(est, "named_steps") else est
    if hasattr(m, "estimators_") and not isinstance(m, VotingRegressor): return f"{len(m.estimators_)} trees"
    if hasattr(m, "n_estimators"): return f"{m.n_estimators} boosting rounds"
    if hasattr(m, "coef_"): return f"linear, {int(np.size(m.coef_))} coefficients"
    if hasattr(m, "tree_"): return f"1 tree, depth {m.get_depth()}"
    return type(m).__name__

fin_rows = []
for name, t in TUNED.items():
    v_runs, t_runs = [], None
    for k in VAL_FOLDS:
        tr, te = FOLDS[k]; _, p, ft, pt = fit_and_forecast(t["est"], tr, HORIZON, t["use_lags"], t["cols"]); v_runs.append({**evaluate(te, p), "fit_s": ft, "predict_s": pt})
    tr, te = FOLDS[TEST_FOLD]; m, p, ft, pt = fit_and_forecast(t["est"], tr, HORIZON, t["use_lags"], t["cols"]); te_m = evaluate(te, p)
    V = pd.DataFrame(v_runs)
    fin_rows.append({"model": name, "feature_set": t["feature_set"], "n_features": len(t["cols"]), "RMSE": V.RMSE.mean(), "RMSE_std": V.RMSE.std(), "WAPE_7d": V.WAPE_7d.mean(),
                     "WAPE_30d": V.WAPE_30d.mean(), "Total90_abs": V.Total90_err.abs().mean(), "test_RMSE": te_m["RMSE"], "test_R2": te_m["R2"], "test_WAPE_30d": te_m["WAPE_30d"],
                     "test_Total90_err": te_m["Total90_err"], "forecast_s": V.predict_s.mean(), "complexity": complexity(m)})
FIN = pd.DataFrame(fin_rows); FIN["score"] = FIN[["RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs"]].rank().mean(axis=1)
FIN = FIN.sort_values(["score", "forecast_s"]).reset_index(drop=True); FIN.to_csv("../docs/final_candidates.csv", index=False)
display(FIN.round(2))
WINNER = FIN.iloc[0].model; W = TUNED[WINNER]
bb = RESULTS["best_baseline"]
print(f"\nSELECTED MODEL: {WINNER}  | feature set {W['feature_set']} | {len(W['cols'])} features | params {tune_log[[t['model'] for t in tune_log].index(WINNER)]['best_params']}")
print(f"Validation RMSE {FIN.iloc[0].RMSE:,.0f} vs best baseline ({bb['model']}) {bb['RMSE']:,.0f}")

# %%
# Does capping outliers in the TRAINING target help? (evaluated on raw sales)
ab = []
for q, lab in [(None, "no capping"), (0.995, "cap at 99.5th pct"), (0.99, "cap at 99th pct (used)"), (0.95, "cap at 95th pct")]:
    s = cv_eval(W["est"], W["use_lags"], W["cols"], cap_q=q); ab.append({"training target": lab, "RMSE": s.RMSE, "WAPE_30d": s.WAPE_30d, "Total90_abs": s.Total90_abs})
AB = pd.DataFrame(ab); display(AB.round(2)); AB.to_csv("../docs/outlier_capping_ablation.csv", index=False)

# %% [markdown]
# ### Test-window evaluation of the selected model
# Now (and only now) we look at the untouched **most recent 90 days**: forecast made with data up to 2014-07-01, compared with what really happened.

# %%
tr, te = FOLDS[TEST_FOLD]; m_t, p_t, _, _ = fit_and_forecast(W["est"], tr, HORIZON, W["use_lags"], W["cols"])
TEST_M = evaluate(te, p_t); RESULTS["test_metrics"] = {k: float(v) for k, v in TEST_M.items()}
bl = baseline_forecasts(tr, HORIZON); bl_m = {n: evaluate(te, f) for n, f in bl.items()}
display(pd.DataFrame({"Selected model": TEST_M, **{n.replace("Baseline: ", "BL: "): v for n, v in bl_m.items()}}).T.round(2))

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].plot(te.index, te.values, color="#9aa5b1", lw=.8, label="actual (daily)"); ax[0].plot(p_t.index, p_t.values, color="#C44E52", lw=2, label="forecast")
ax[0].set_title("Test window – daily"); ax[0].legend()
wk_a = te.groupby(np.arange(len(te)) // 7).sum(); wk_f = p_t.groupby(np.arange(len(te)) // 7).sum()
ax[1].bar(wk_a.index - .2, wk_a.values, .4, label="actual", color="#4C72B0"); ax[1].bar(wk_f.index + .2, wk_f.values, .4, label="forecast", color="#C44E52")
ax[1].set_title("Test window – weekly totals"); ax[1].set_xlabel("week of the forecast"); ax[1].legend()
res = te - p_t; sns.histplot(res, bins=40, ax=ax[2], color="#8172B3"); ax[2].set_title("Daily forecast errors (actual − forecast)")
plt.tight_layout(); savefig("15_test_window"); plt.show()
print(f"Actual 90-day total: {te.sum():,.0f} | Forecast: {p_t.sum():,.0f} | error {TEST_M['Total90_err']:+.1f}%")

# %%
# Back-test of the selected model on all four windows (shows consistency across time)
fig, ax = plt.subplots(1, 4, figsize=(18, 3.6), sharey=False); RES_ALL = []
for i, k in enumerate([4, 3, 2, 1]):
    tr_k, te_k = FOLDS[k]; _, p_k, _, _ = fit_and_forecast(W["est"], tr_k, HORIZON, W["use_lags"], W["cols"]); RES_ALL.append((te_k, p_k))
    a = te_k.groupby(np.arange(len(te_k)) // 7).sum(); f = p_k.groupby(np.arange(len(te_k)) // 7).sum()
    ax[i].bar(a.index - .2, a.values, .4, color="#4C72B0", label="actual"); ax[i].bar(f.index + .2, f.values, .4, color="#C44E52", label="forecast")
    ax[i].set_title(f"{'TEST' if k == 1 else 'Validation'} {te_k.index.min():%b %Y} – {te_k.index.max():%b %Y}\nweekly totals", fontsize=10)
ax[0].legend(); plt.tight_layout(); savefig("16_backtest_all_windows"); plt.show()

# %%
# Honest check: 90-day TOTAL error (forecast vs actual, %) of the selected model and the baselines in each of the four windows
rows = []
for (te_k, p_k), k in zip(RES_ALL, [4, 3, 2, 1]):
    bl_k = baseline_forecasts(FOLDS[k][0], HORIZON)
    r = {"window": f"{te_k.index.min():%d %b %Y} – {te_k.index.max():%d %b %Y}", "role": "TEST" if k == 1 else "validation", "actual_total": te_k.sum(),
         "Selected model": (p_k.sum() / te_k.sum() - 1) * 100}
    for n, f in bl_k.items(): r[n.replace("Baseline: ", "BL: ")] = (f.sum() / te_k.sum() - 1) * 100
    rows.append(r)
WIN = pd.DataFrame(rows).set_index("window"); display(WIN.round(1))
val_abs = WIN[WIN.role == "validation"].drop(columns=["role", "actual_total"]).abs().mean(); best_bl = val_abs.drop("Selected model").idxmin()
print(f"Average |90-day total error| on the 3 validation windows: selected model {val_abs['Selected model']:.1f}%  vs best baseline ({best_bl}) {val_abs[best_bl]:.1f}%")
print(f"Latest (TEST) window: selected model {WIN.loc[WIN.role == 'TEST', 'Selected model'].iloc[0]:+.1f}%  vs {best_bl} {WIN.loc[WIN.role == 'TEST', best_bl].iloc[0]:+.1f}%")
y13 = daily.loc["2013-07-02":"2013-09-29"].sum(); y14 = daily.loc["2014-07-02":"2014-09-29"].sum()
print(f"Context: sales in the test window were {y14 / y13 - 1:+.0%} versus the same weeks of 2013, while Jan–Sep 2014 overall was "
      f"{daily.loc['2014-01-01':'2014-09-29'].sum() / daily.loc['2013-01-01':'2013-09-29'].sum() - 1:+.0%} versus Jan–Sep 2013.")

# %% [markdown]
# **Honest verdict on the test window.** The model was chosen on the validation windows, and the test window is where it is checked. The table above shows
# that the **selected model is competitive with the best benchmark on the validation windows, but over-forecast the most recent window by a large margin** –
# that window was an unusually weak quarter (sales fell sharply versus the same weeks of the previous year, printed above), while the model extrapolates the
# long-term growth trend. A model that extrapolates growth is exposed to exactly this risk; the benchmark that only repeats the recent average is not, but it
# cannot see the yearly seasonal pattern and under-forecast strong quarters (first window). **We therefore treat the forecast as a baseline planning scenario
# and always show its wide uncertainty band**, and the model should be re-trained whenever new data arrives.
#
# **Observation.** Weekly totals are very erratic (a week can be 0 or 35k) – the forecast follows the *typical level and seasonal shape*,
# not individual bursts. This is the honest limit of what the data can support; the planning-grade information is in the
# **30-day and 90-day totals** and in the uncertainty band computed next.

# %% [markdown]
# ## 13. Final training and the 90-day forecast
# The selected configuration is re-trained on **all available history** and used to forecast the next 90 days
# (the day after the last date in the data, 2014-09-30, to 2014-12-28).
# Uncertainty bands are built from the **real out-of-sample errors** of the four back-test windows (5 % – 95 % quantiles of the errors).

# %%
FINAL_MODEL, FC, fit_t, pred_t = fit_and_forecast(W["est"], daily, HORIZON, W["use_lags"], W["cols"])
CAP = float(daily.quantile(.99))
err_d = pd.concat([a - b for a, b in RES_ALL])
def block_err(n):
    out = []
    for a, b in RES_ALL:
        d = (a - b); out += [d.iloc[i:i + n].sum() for i in range(0, len(d) - n + 1, n)]
    return np.array(out)
INTERVALS = {"daily": {"q05": float(err_d.quantile(.05)), "q95": float(err_d.quantile(.95))},
             "weekly": {"q05": float(np.quantile(block_err(7), .05)), "q95": float(np.quantile(block_err(7), .95))},
             "30day": {"q05": float(np.quantile(block_err(30), .05)), "q95": float(np.quantile(block_err(30), .95))},
             "90day": {"q05": float(np.quantile(block_err(90), .05)), "q95": float(np.quantile(block_err(90), .95))}}
print("Error quantiles (actual − forecast) from back-tests:"); display(pd.DataFrame(INTERVALS).T.round(0))
fc_df = FC.to_frame("forecast"); fc_df["lower"] = (fc_df.forecast + INTERVALS["daily"]["q05"]).clip(lower=0); fc_df["upper"] = fc_df.forecast + INTERVALS["daily"]["q95"]
print(f"Forecast period: {FC.index.min().date()} → {FC.index.max().date()} | total = {FC.sum():,.0f} | avg per day = {FC.mean():,.0f} | inference time = {pred_t * 1000:.0f} ms")
tot = FC.sum(); print(f"90-day total: {tot:,.0f}  (5-95% range {max(tot + INTERVALS['90day']['q05'], 0):,.0f} – {tot + INTERVALS['90day']['q95']:,.0f})")
fc_df.head(7).round(0)

# %%
fig, ax = plt.subplots(2, 1, figsize=(14, 8.5))
hist = daily.iloc[-180:]
ax[0].plot(hist.index, hist.values, color="#9aa5b1", lw=.8, label="actual daily sales"); ax[0].plot(hist.rolling(14).mean(), color="#1f3b73", lw=1.8, label="14-day avg (actual)")
ax[0].plot(fc_df.index, fc_df.forecast, color="#C44E52", lw=2.2, label="forecast (expected daily sales)")
ax[0].fill_between(fc_df.index, fc_df.lower, fc_df.upper.clip(upper=hist.max()), color="#C44E52", alpha=.12, label="5–95 % error band")
ax[0].axvline(daily.index.max(), color="k", ls=":"); ax[0].set_title("Daily sales: last 180 days and the 90-day forecast"); ax[0].legend(loc="upper left", fontsize=8)
wk = FC.resample("W-SUN").sum(); ax[1].bar(wk.index, wk.values, width=5, color="#C44E52"); ax[1].set_title("Forecast – weekly totals")
plt.tight_layout(); savefig("17_final_forecast"); plt.show()
monthly_fc = FC.groupby(FC.index.to_period("M")).sum(); print("Forecast by calendar month:", monthly_fc.round(0).to_dict())
ly = daily.reindex(FC.index - pd.Timedelta(days=364)).sum(); print(f"Same 90 days one year earlier (actual): {ly:,.0f}  → forecast is {tot / ly - 1:+.0%} vs last year")
ytd = daily.loc['2014-01-01':'2014-09-29'].sum() / daily.loc['2013-01-01':'2013-09-29'].sum() - 1
print(f"Reasonableness check: year-to-date 2014 is {ytd:+.0%} vs 2013, the forecast for the next 90 days is {tot / ly - 1:+.0%} vs the same days of 2013 "
      f"(a deliberately cautious level compared with the year-to-date growth).")

# %%
# Feature importance of the final model (permutation importance on the last 365 days)
from sklearn.inspection import permutation_importance
capped = daily.clip(upper=CAP); Xi = fc.build_frame(capped, W["use_lags"], ORIGIN)[W["cols"]].iloc[-365:]; yi = capped.loc[Xi.index]
pi = permutation_importance(FINAL_MODEL, Xi, yi, n_repeats=8, random_state=SEED, scoring="neg_root_mean_squared_error")
IMP = pd.Series(pi.importances_mean, index=W["cols"]).sort_values(ascending=False)
IMP_PCT = (IMP.clip(lower=0) / IMP.clip(lower=0).sum() * 100) if IMP.clip(lower=0).sum() > 0 else IMP * 0
IMP_PCT.head(12).iloc[::-1].plot.barh(figsize=(8, 5), color="#4C72B0"); plt.title("Most important features of the final model (permutation importance, %)")
plt.tight_layout(); savefig("18_feature_importance"); plt.show(); display(IMP_PCT.head(10).round(1).to_frame("importance %"))

# %% [markdown]
# ## 14. Save the model (joblib) with versioning
# Artifacts written to `api/artifacts/`: `model.joblib` (fitted pipeline), `metadata.json` (version, features, metrics, error bands, importances, leaderboard),
# `history.csv` (daily sales – needed by the API for lag features and charts) and `forecast.csv` (the pre-computed 90-day forecast).

# %%
ART = "../api/artifacts"
joblib.dump(FINAL_MODEL, f"{ART}/model.joblib", compress=3)
daily.to_frame("sales").to_csv(f"{ART}/history.csv")
fc_df.to_csv(f"{ART}/forecast.csv", index_label="date")
data_hash = hashlib.sha256(open(DATA, "rb").read()).hexdigest()[:12]
best_p = tune_log[[t["model"] for t in tune_log].index(WINNER)]["best_params"]
META = {
 "model_version": "1.0.0", "trained_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", "data_hash": data_hash,
 "model_name": WINNER, "model_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_p.items()},
 "feature_set": W["feature_set"], "use_lags": bool(W["use_lags"]), "features": W["cols"], "origin": str(ORIGIN.date()),
 "cap_value": CAP, "horizon_days": HORIZON, "history_start": str(daily.index.min().date()), "history_end": str(daily.index.max().date()),
 "forecast_start": str(FC.index.min().date()), "forecast_end": str(FC.index.max().date()), "n_training_days": int(len(daily)),
 "validation_metrics": {k: float(FIN.iloc[0][k]) for k in ["RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs"]},
 "test_metrics": RESULTS["test_metrics"], "best_baseline": {k: (float(v) if not isinstance(v, str) else v) for k, v in RESULTS["best_baseline"].items()},
 "intervals": INTERVALS, "feature_importance": {k: float(v) for k, v in IMP_PCT.head(15).items()},
 "leaderboard": cand[["model", "family", "feature_set", "RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs", "score"]].head(15).round(3).to_dict("records"),
 "n_models_compared": RESULTS["n_models"], "inference_seconds_90d": float(pred_t),
 "window_errors": WIN.reset_index().round(2).to_dict("records"),
}
json.dump(META, open(f"{ART}/metadata.json", "w"), indent=2)
print("Saved:", os.listdir(ART), "| model size:", f"{os.path.getsize(f'{ART}/model.joblib') / 1024:.0f} KB")

# %%
# Sanity check: re-load from disk exactly like the API does and reproduce the forecast
m2 = joblib.load(f"{ART}/model.joblib"); hist2 = pd.read_csv(f"{ART}/history.csv", index_col=0, parse_dates=True)["sales"]
re = fc.forecast(m2, hist2.clip(upper=META["cap_value"]), HORIZON, META["use_lags"], META["features"], pd.Timestamp(META["origin"]), max_value=META["cap_value"])
print("Max difference between notebook forecast and re-loaded model forecast:", float((re - FC).abs().max()))

# %%
json.dump({k: v for k, v in RESULTS.items() if k != "feature_ranking"} | {"winner": WINNER, "tune_log": tune_log,
          "final_candidates": FIN.round(3).to_dict("records"), "ablation": AB.round(3).to_dict("records"),
          "forecast_total": float(FC.sum()), "forecast_vs_last_year": float(tot / ly - 1) if ly else None, "last_year_total": float(ly),
          "monthly_forecast": {str(k): float(v) for k, v in monthly_fc.items()}, "feature_ranking": RESULTS["feature_ranking"][:12],
          "feature_set_means": cand.groupby("feature_set")[["RMSE", "WAPE_7d", "Total90_abs"]].mean().round(2).to_dict("index"),
          "baselines_val": base[["model", "RMSE", "R2", "WAPE_7d", "WAPE_30d", "Total90_err"]].round(2).to_dict("records"),
          "test_baselines": {n: {k: float(v) for k, v in m.items()} for n, m in bl_m.items()}, "top15": cand[show].head(15).round(2).to_dict("records")},
          open("../docs/results_summary.json", "w"), indent=2, default=str)

# %% [markdown]
# ## 15. Deployment (summary)
# * **API** – `api/main.py` (FastAPI): `/health`, `/model-info`, `/forecast`, `/forecast/date`, `/history`, `/leaderboard`; input validation, error handling, env-var configuration, versioned model.
# * **Docker** – `api/Dockerfile`; **Render** – `render.yaml` (web service from the Dockerfile).
# * **UI** – `frontend/app.py` (Streamlit) talks to the API through the `API_URL` environment variable.
# * Full commands are in `README.md`.
