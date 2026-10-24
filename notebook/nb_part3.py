# %% [markdown]
# ## 7. Train / Test Split (time-based, rolling-origin)
#
# A random split would leak the future into training. Instead we **cut the timeline** and repeatedly ask the model the *exact*
# question the business will ask: *"given everything up to day D, forecast the next 90 days."*
#
# * **Fold 1 – final TEST window** = the most recent 90 days (2014-07-02 → 2014-09-29). Never used for model choice or tuning.
# * **Folds 2, 3, 4 – VALIDATION windows** = the three 90-day windows before it; each is predicted by a model trained only on the data before that window
#   (expanding window). Used to compare models and tune hyper-parameters.
#
# One 90-day window of such noisy data would be a lucky-or-unlucky coin toss; averaging three validation windows is much more reliable.

# %%
def make_folds(series, H=HORIZON, n_folds=4):
    folds, n = {}, len(series)
    for k in range(1, n_folds + 1):
        end = n - (k - 1) * H; start = end - H
        folds[k] = (series.iloc[:start], series.iloc[start:end])
    return folds
FOLDS = make_folds(daily)
for k, (tr, te) in FOLDS.items():
    role = "TEST      " if k == 1 else "VALIDATION"
    print(f"Fold {k} [{role}] train: {tr.index.min().date()} → {tr.index.max().date()} ({len(tr)} d) | predict: {te.index.min().date()} → {te.index.max().date()} ({len(te)} d)")
VAL_FOLDS, TEST_FOLD = [2, 3, 4], 1

fig, ax = plt.subplots(figsize=(13, 2.8))
for k, (tr, te) in FOLDS.items():
    ax.barh(k, (tr.index.max() - tr.index.min()).days, left=tr.index.min(), color="#4C72B0", height=.5)
    ax.barh(k, HORIZON, left=te.index.min(), color="#C44E52" if k == 1 else "#DD8452", height=.5)
ax.set_yticks(list(FOLDS)); ax.set_yticklabels([f"Fold {k}" for k in FOLDS]); ax.invert_yaxis()
ax.set_title("Rolling-origin back-test: blue = training, orange = validation window, red = final test window")
plt.tight_layout(); savefig("12_split_folds"); plt.show()

# %% [markdown]
# ## 8. Evaluation metrics for this regression problem
# Classification metrics (confusion matrix, accuracy, F1, ROC-AUC) do **not** apply because the target is a continuous amount.
# The regression equivalents we use:
#
# | Metric | Meaning | Note |
# |---|---|---|
# | **RMSE** | typical daily error in sales units, penalises big misses | main daily metric |
# | **MAE** | average absolute daily error | reported, but *not* used for ranking: on a series where 60 % of days are 0, the MAE-optimal prediction is ≈ 0, which is useless for revenue planning |
# | **R²** | share of daily variance explained (0 = no better than predicting the test-window mean) | will be low – daily noise dominates |
# | **WAPE-7d / WAPE-30d** | error on *weekly* / *30-day* totals ÷ actual total | how useful the forecast is for planning |
# | **Total-90d error %** | (forecast total − actual total) ÷ actual total over the 90 days | the "quarterly revenue" error; sign shows over/under-forecast |
#
# MAPE is deliberately **not** used (actual = 0 on most days → division by zero).
# Models are **ranked** by the average rank on RMSE, WAPE-7d, WAPE-30d and |Total-90d error| (all consistent with predicting the *expected* sales).

# %%
def evaluate(y, p):
    p = pd.Series(np.asarray(p, dtype=float), index=y.index); e = y - p
    blk7 = np.arange(len(y)) // 7; m7 = blk7 < len(y) // 7
    blk30 = np.arange(len(y)) // 30; m30 = blk30 < len(y) // 30
    w_y, w_p = y[m7].groupby(blk7[m7]).sum(), p[m7].groupby(blk7[m7]).sum()
    m_y, m_p = y[m30].groupby(blk30[m30]).sum(), p[m30].groupby(blk30[m30]).sum()
    return {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
            "WAPE_7d": (w_y - w_p).abs().sum() / w_y.sum() * 100,
            "WAPE_30d": (m_y - m_p).abs().sum() / m_y.sum() * 100,
            "Total90_err": (p.sum() - y.sum()) / y.sum() * 100}

# %%
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
ORIGIN = daily.index.min()

def fit_and_forecast(est, train, H, use_lags, cols=None, cap_q=0.99):
    """Cap outliers (training only) -> build features -> fit -> forecast H days."""
    cap = train.quantile(cap_q) if cap_q else np.inf
    trc = train.clip(upper=cap) if cap_q else train
    cols = list(cols) if cols is not None else fc.feature_names(use_lags)
    X = fc.build_frame(trc, use_lags, ORIGIN)[cols]; y = trc.loc[X.index]
    m = clone(est); t0 = time.time(); m.fit(X, y); fit_t = time.time() - t0
    t0 = time.time(); p = fc.forecast(m, trc, H, use_lags, cols, ORIGIN, max_value=(cap if cap_q else None)); pred_t = time.time() - t0
    return m, p, fit_t, pred_t

def baseline_forecasts(train, H):
    idx = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    ly = train.reindex(idx - pd.Timedelta(days=364)).fillna(0).values
    return {"Baseline: mean of all history": pd.Series(train.mean(), idx),
            "Baseline: mean of last 365 days": pd.Series(train.iloc[-365:].mean(), idx),
            "Baseline: mean of last 90 days": pd.Series(train.iloc[-90:].mean(), idx),
            "Baseline: seasonal naive (same day last year)": pd.Series(ly, idx)}

# %% [markdown]
# ## 9. Build and compare **all** regression models
# Besides Linear and Ridge (required by the assignment) we train **every regression family available** in scikit-learn, plus XGBoost,
# LightGBM and CatBoost – 36 models: linear, regularised, robust, generalised-linear, kernel/neighbour, tree, ensemble, boosting, neural network.
# Each model is tested on **both feature sets** (A: calendar, B: calendar + lags) → 72 combinations × 4 folds.
# (`GammaRegressor` is the only sklearn regressor left out: it requires strictly positive targets and 60 % of our days are 0.)

# %%
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, LassoLars, OrthogonalMatchingPursuit, BayesianRidge,
    ARDRegression, HuberRegressor, TheilSenRegressor, RANSACRegressor, SGDRegressor, PassiveAggressiveRegressor,
    PoissonRegressor, TweedieRegressor, QuantileRegressor)
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, VotingRegressor)
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def pipe(est, scale=True):
    return Pipeline([("scale", StandardScaler() if scale else "passthrough"), ("model", est)])

def build_zoo():
    z = {}   # name -> (pipeline, family)
    add = lambda n, e, fam, scale=True: z.__setitem__(n, (pipe(e, scale), fam))
    # --- linear family
    add("Linear Regression", LinearRegression(), "Linear")
    add("Ridge", Ridge(alpha=100), "Linear")
    add("Lasso", Lasso(alpha=20, max_iter=20000), "Linear")
    add("ElasticNet", ElasticNet(alpha=20, l1_ratio=.5, max_iter=20000), "Linear")
    add("Lasso-LARS", LassoLars(alpha=20), "Linear")
    add("Orthogonal Matching Pursuit", OrthogonalMatchingPursuit(n_nonzero_coefs=8), "Linear")
    add("Bayesian Ridge", BayesianRidge(), "Linear")
    add("ARD Regression", ARDRegression(), "Linear")
    add("PLS Regression", PLSRegression(n_components=5), "Linear")
    # --- robust / online / generalised linear
    add("Huber", HuberRegressor(alpha=10, max_iter=500), "Robust / GLM")
    add("Theil-Sen", TheilSenRegressor(random_state=SEED, max_subpopulation=2000), "Robust / GLM")
    add("RANSAC", RANSACRegressor(random_state=SEED), "Robust / GLM")
    add("SGD Regressor", SGDRegressor(alpha=1.0, max_iter=2000, tol=1e-3, random_state=SEED), "Robust / GLM")
    add("Passive-Aggressive", PassiveAggressiveRegressor(C=.01, max_iter=1000, random_state=SEED), "Robust / GLM")
    add("Poisson GLM", PoissonRegressor(alpha=1.0, max_iter=1000), "Robust / GLM")
    add("Tweedie GLM (p=1.5)", TweedieRegressor(power=1.5, alpha=1.0, link="log", max_iter=1000), "Robust / GLM")
    add("Quantile (median)", QuantileRegressor(quantile=.5, alpha=1e-3, solver="highs"), "Robust / GLM")
    # --- kernel / neighbours / GP
    add("KNN", KNeighborsRegressor(n_neighbors=25), "Kernel / Neighbours")
    add("SVR (RBF)", SVR(C=2000, epsilon=50), "Kernel / Neighbours")
    add("NuSVR", NuSVR(C=2000, nu=.5), "Kernel / Neighbours")
    add("Linear SVR", LinearSVR(C=100, epsilon=50, max_iter=20000, random_state=SEED), "Kernel / Neighbours")
    add("Kernel Ridge (RBF)", KernelRidge(alpha=10, kernel="rbf", gamma=.02), "Kernel / Neighbours")
    add("Gaussian Process", GaussianProcessRegressor(kernel=C(1.0) * RBF(5.0) + WhiteKernel(1.0), normalize_y=True, random_state=SEED), "Kernel / Neighbours")
    # --- trees & ensembles
    add("Decision Tree", DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=SEED), "Tree / Ensemble", False)
    add("Extra Tree (single)", ExtraTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=SEED), "Tree / Ensemble", False)
    add("Random Forest", RandomForestRegressor(300, min_samples_leaf=10, random_state=SEED, n_jobs=1), "Tree / Ensemble", False)
    add("Extra Trees", ExtraTreesRegressor(300, min_samples_leaf=10, random_state=SEED, n_jobs=1), "Tree / Ensemble", False)
    add("Bagging (trees)", BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=10), n_estimators=100, random_state=SEED, n_jobs=1), "Tree / Ensemble", False)
    add("AdaBoost", AdaBoostRegressor(n_estimators=100, learning_rate=.05, random_state=SEED), "Tree / Ensemble", False)
    # --- boosting
    add("Gradient Boosting", GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=.05, subsample=.8, random_state=SEED), "Boosting", False)
    add("HistGradient Boosting", HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=150, min_samples_leaf=20, random_state=SEED), "Boosting", False)
    add("XGBoost", XGBRegressor(n_estimators=250, learning_rate=.03, max_depth=3, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=SEED), "Boosting", False)
    add("LightGBM", LGBMRegressor(n_estimators=200, learning_rate=.03, num_leaves=8, min_child_samples=20, verbose=-1, n_jobs=1, random_state=SEED), "Boosting", False)
    add("CatBoost", CatBoostRegressor(iterations=250, depth=4, learning_rate=.05, verbose=0, thread_count=1, random_seed=SEED), "Boosting", False)
    # --- neural net + voting ensemble
    add("MLP (neural net)", MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=400, random_state=SEED), "Neural / Ensemble")
    add("Voting (Ridge+ExtraTrees+GBM+LGBM)", VotingRegressor([("r", pipe(Ridge(alpha=100))), ("e", ExtraTreesRegressor(200, min_samples_leaf=10, random_state=SEED)),
         ("g", GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=.05, subsample=.8, random_state=SEED)),
         ("l", LGBMRegressor(n_estimators=200, learning_rate=.03, num_leaves=8, verbose=-1, n_jobs=1, random_state=SEED))]), "Neural / Ensemble", False)
    return z
ZOO = build_zoo(); print(len(ZOO), "regression models:"); print(", ".join(ZOO))

# %%
# ---- Run the full back-test: every model x 2 feature sets x 4 folds (≈ 3–5 min on one CPU core) --------------------
rows, t_start = [], time.time()
for fold, (tr, te) in FOLDS.items():                                   # baselines (not candidates, only reference points)
    for bn, bp in baseline_forecasts(tr, HORIZON).items():
        rows.append({"model": bn, "family": "Baseline", "feature_set": "-", "fold": fold, **evaluate(te, bp), "fit_s": 0, "predict_s": 0})
fails = {}
for name, (est, fam) in ZOO.items():
    for use_lags, fs in [(False, "A: calendar"), (True, "B: calendar + lags")]:
        for fold, (tr, te) in FOLDS.items():
            try:
                _, p, ft, pt = fit_and_forecast(est, tr, HORIZON, use_lags)
                rows.append({"model": name, "family": fam, "feature_set": fs, "fold": fold, **evaluate(te, p), "fit_s": ft, "predict_s": pt})
            except Exception as ex:
                fails[(name, fs)] = str(ex)[:80]
    print(f"{name:<38} done  ({time.time() - t_start:5.0f}s elapsed)")
BT = pd.DataFrame(rows); BT.to_csv("../docs/backtest_all_folds.csv", index=False)
print("\nFailed combinations:", fails if fails else "none")

# %%
# ---- Summarise: validation mean (folds 2-4) and final test (fold 1) --------------------------------------------------------
KEY = ["model", "family", "feature_set"]
val = BT[BT.fold.isin(VAL_FOLDS)].groupby(KEY)[["MAE", "RMSE", "R2", "WAPE_7d", "WAPE_30d", "Total90_err", "fit_s", "predict_s"]].mean()
val["Total90_abs"] = BT[BT.fold.isin(VAL_FOLDS)].assign(a=lambda d: d.Total90_err.abs()).groupby(KEY)["a"].mean()
val["RMSE_std"] = BT[BT.fold.isin(VAL_FOLDS)].groupby(KEY)["RMSE"].std()
test = BT[BT.fold == TEST_FOLD].set_index(KEY)[["MAE", "RMSE", "R2", "WAPE_7d", "WAPE_30d", "Total90_err"]].add_prefix("test_")
LB = val.join(test).reset_index()
rank_cols = ["RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs"]
cand = LB[LB.family != "Baseline"].copy()
cand["score"] = cand[rank_cols].rank().mean(axis=1)              # average rank: lower = better
cand = cand.sort_values("score").reset_index(drop=True); cand.index += 1
base = LB[LB.family == "Baseline"].sort_values("RMSE")
cand.to_csv("../docs/model_leaderboard.csv"); base.to_csv("../docs/baselines.csv", index=False)
show = ["model", "feature_set", "RMSE", "WAPE_7d", "WAPE_30d", "Total90_abs", "score", "test_RMSE", "test_Total90_err"]
print("TOP 15 candidates (validation = mean of folds 2-4; test = fold 1; score = average rank, lower is better)")
display(cand[show].head(15).round(2))
print("Reference baselines (validation mean):"); display(base[["model", "RMSE", "R2", "WAPE_7d", "WAPE_30d", "Total90_err"]].round(2))

# %%
# ---- Visual comparison of all models ----------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(16, 10), gridspec_kw={"width_ratios": [1.3, 1]})
order = cand.groupby("model").score.min().sort_values().index[::-1]
pv = cand.pivot_table(index="model", columns="feature_set", values="RMSE").loc[order]
pv.plot.barh(ax=ax[0], width=.8, color=["#4C72B0", "#DD8452"]); ax[0].set_title("Validation RMSE (daily) – all models, both feature sets")
ax[0].axvline(base.RMSE.min(), color="red", ls="--", lw=1.2); ax[0].text(base.RMSE.min(), -0.9, " best baseline", color="red", fontsize=8)
ax[0].set_xlim(left=pv.min().min() * .9); ax[0].set_xlabel("RMSE (lower = better)")
sc = ax[1].scatter(cand.RMSE, cand.Total90_abs, c=cand.score, cmap="viridis_r", s=45)
for _, r in cand.head(6).iterrows(): ax[1].annotate(r.model, (r.RMSE, r.Total90_abs), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax[1].set_xlabel("Validation RMSE (daily)"); ax[1].set_ylabel("Validation |90-day total error| %"); ax[1].set_title("Accuracy trade-off (colour = composite score)")
plt.colorbar(sc, ax=ax[1], label="average rank score"); plt.tight_layout(); savefig("13_model_comparison"); plt.show()

# %%
# ---- Feature-set and family summary ------------------------------------------------------------------------------------------
print("Average validation RMSE by feature set (all models):"); display(cand.groupby("feature_set")[["RMSE", "WAPE_7d", "Total90_abs"]].mean().round(2))
print("Best combination per model family:"); display(cand.sort_values("score").groupby("family").head(1)[["family", "model", "feature_set", "RMSE", "WAPE_30d", "Total90_abs", "score"]].round(2))
RESULTS["n_models"] = len(ZOO); RESULTS["n_combos"] = int(len(cand)); RESULTS["best_baseline"] = base.iloc[0][["model", "RMSE", "WAPE_7d", "WAPE_30d", "Total90_err"]].to_dict()
