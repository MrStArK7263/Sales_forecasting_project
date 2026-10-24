# Sales Forecasting: 90-Day Daily Forecast – Project Documentation

*Summary of `Sales_Forecasting.ipynb`: what was done, which models were used and why.*

## 1. Summary

- **Goal:** forecast total company sales for every day of the next 90 days (30 Sep 2014 → 28 Dec 2014) with regression models.
- **Data:** 2,172 order lines (33 columns) from 21 Jan 2011 to 29 Sep 2014, aggregated to **1,348 daily values**.
- **Models:** **36 regression models** (every scikit-learn regressor family plus XGBoost, LightGBM, CatBoost) were compared on two feature sets, with time-based back-testing, against four simple baselines.
- **Selected model:** **Lasso regression (alpha = 0.5) on calendar features.** Linear models with regularisation ranked first; complex models (random forest, boosting, neural network) ranked lower because daily sales are mostly noise.
- **Forecast:** about **213,200** in total for the next 90 days (about 2,370 per day), likely range **122,600 – 286,300**, which is 12 % below the same 90 days of 2013.
- **Key caveat (honest result):** daily sales are spiky (60 % of days have no orders), so the daily R² is about 0 for every model. The selected model beats the best benchmark on the validation windows (90-day total error 14.5 % vs 19.5 %), **but over-forecast the most recent window by 71 %**, because that quarter was unusually weak. Use the forecast as a planning scenario together with its range.

## 2. Problem and data

| Item | Value |
|---|---|
| Problem type | Regression, time series (target = total `Sales` per day) |
| Raw data | one row per order line: product, quantity, Sales, Discount, COS, GP, customer, country, employee, dates… |
| Rows / columns | 2,172 / 33 (19 numeric, 14 text/date) |
| Period | 21 Jan 2011 → 29 Sep 2014 (1,348 days) |
| Days with orders | 533 (39.5 %); **815 days (60.5 %) have zero sales** |
| Target construction | sum of `Sales` per calendar day; days without orders set to 0 so the calendar is complete |
| Forecast horizon | 90 days, daily |

## 3. Approach (follows the assignment steps)

| Step | What was done | Why |
|---|---|---|
| 1 Data understanding | shapes, dtypes, numeric/categorical split, statistics, target distribution | know the data before modelling |
| 2 Cleaning | missing, duplicates, types, invalid values, categories, outliers, unused columns, leakage | protect against wrong totals and unrealistic accuracy |
| 3 EDA | trend, seasonality, weekday, box plots, count plots, category/country/product, feature vs target, outliers | find what drives sales |
| 4 Correlation | heatmaps of raw and engineered features, multicollinearity | find leakage and redundant columns |
| 5 Feature engineering | calendar, cyclical and lag/rolling features | give the model information available for future dates |
| 6 Feature selection | correlation, mutual information, random-forest importance; before vs after | simpler, less over-fitting |
| 7 Split | rolling-origin time split: 1 test + 3 validation windows of 90 days | never train on the future |
| 8–9 Models + evaluation | 36 models × 2 feature sets × 4 windows + 4 baselines; regression metrics | find the best model fairly |
| 10 Tuning | grid search on validation windows; before vs after | improve the best candidates |
| 11 Final selection | multi-criteria rule (accuracy, stability, speed, complexity) | highest accuracy alone is not enough |
| 12–16 Save, API, UI, deploy | joblib, FastAPI, Streamlit, Docker, Render | production requirements |

## 4. Data cleaning decisions

| Check | Finding | Decision and reason |
|---|---|---|
| Missing values | only `ReportsTo` (6.7 %), `Fax` (36 %), `Phone` (11 %), `PostalCode` (24 %); `OrderDate` and `Sales` complete | the gaps are in contact columns that do not affect total sales, so they are not used; no imputation needed |
| Duplicates | 0 duplicate rows, 0 duplicate (Order, Product) keys | nothing to remove (duplicates would double-count revenue) |
| Data types | `OrderDate` was text with a UTC suffix; `PostalCode` numeric | converted to dates (time-zone removed); postal code is an identifier and is not used |
| Invalid values | no Sales/Quantity ≤ 0, no negative Discount/COS/GP/Freight; `Sales = COS + GP` to rounding | data is consistent |
| Categories | 8 consistent categories, 21 countries = 21 country codes | no fixes needed (only a typographic accent in names like *Men´s Clothes*) |
| Outliers | 14 days above the 99th percentile (about 15,990); the maximum day is about 60,130 | real big orders, so **kept**; for training only, the target is **capped at the 99th percentile**; models are always evaluated on raw sales |
| Unnecessary columns | customer contacts, employee details, geo coordinates, IDs | not usable for a total-sales forecast of future dates |
| Leakage | Quantity, Discount, COS, GP, Freight and number of orders are same-day outcomes of sales | **excluded**; only calendar information and *past* sales are allowed |

Capping check on the validation windows: no capping RMSE 4,612 and 90-day total error 17.0 %; cap at the 99.5th percentile 4,603 and 15.0 %; **cap at the 99th percentile (used) 4,607 and 14.5 %**; cap at the 95th percentile 4,640 and 24.7 %. Capping at the 99th percentile gives the most reliable totals.

## 5. EDA – key findings

- **60 % of days have zero sales** and the rest are right-skewed. This makes daily forecasting inherently noisy and rules out MAPE.
- **Strong growth** from 2011 to 2013–2014 (not stationary); sales arrive in bursts.
- **Yearly seasonality:** averaged over 2012–2013, **December (about 100,900), September (74,200) and July (58,500)** are the strongest months; **November (19,600), May (25,300) and February (27,500)** the weakest.
- **Weekday effect is weak:** average daily sales range only from 1,072 (Saturday) to 1,552 (Monday).
- **Concentration:** the top 3 countries make 49 % of sales; the top 10 products make 52 %.
- Same-day quantity, number of orders and profit follow sales almost linearly, which is exactly why they are leakage.

## 6. Correlation analysis

- On daily aggregates: Sales–COS **1.00**, Sales–GP **0.95**, COS–GP **0.92**, Freight–order lines **0.92**; no negative and no near-zero correlations. These columns are leakage and multicollinear, and are removed.
- On engineered features: every feature is only weakly correlated with the noisy daily target (strongest is the trend counter `t`, |r| = 0.25). **Eight feature pairs have |r| > 0.9** (for example month ↔ day-of-year ↔ week-of-year ↔ quarter), so linear models need regularisation, while tree models are not affected.

## 7. Feature engineering and selection

| Group | Features |
|---|---|
| Trend | `t` = days since first date |
| Calendar | day of week, weekend flag, day of month, week of year, month, quarter, day of year, month-start / month-end / quarter-end, December flag |
| Cyclical | sine and cosine of day-of-year (two harmonics), weekday and month |
| Lags / rolling (Set B only) | sales 1, 7, 14, 28 days ago; rolling mean over 7/14/28/90 days; 28-day std; share of days with orders; days since last sale (all shifted by one day, so no leakage) |

Two feature sets were compared: **Set A = calendar only (20 features)** can predict all 90 days directly; **Set B = calendar + lags (31 features)** needs recursive forecasting (each predicted day is fed back).

**Selection before vs after** (validation, top candidates): keeping the top 16 of 20 features gave the same accuracy as all 20 (Ridge RMSE 4,631.2 → 4,631.6) and top 8 was slightly worse (4,650), so only light pruning helps. Lasso kept all features and Ridge kept the top 16, chosen by the same 4-metric average-rank rule.

## 8. Validation design

A random split would leak the future. A **rolling-origin back-test** with 90-day windows was used; each window is predicted by a model trained only on earlier data.

| Window | Role | Predicted period |
|---|---|---|
| Fold 4 | validation | 05 Oct 2013 – 02 Jan 2014 |
| Fold 3 | validation | 03 Jan 2014 – 02 Apr 2014 |
| Fold 2 | validation | 03 Apr 2014 – 01 Jul 2014 |
| **Fold 1** | **final test (never used for choices)** | 02 Jul 2014 – 29 Sep 2014 |

## 9. Evaluation metrics (regression)

Classification metrics (confusion matrix, accuracy, F1, ROC-AUC) do not apply to a continuous target. Used instead: **RMSE** (daily error), **MAE** (reported only: with 60 % zero days the MAE-optimal forecast is about 0), **R²**, **WAPE on weekly and 30-day totals**, and **90-day total error %**. Models are **ranked by the average rank** of RMSE, WAPE-7d, WAPE-30d and |90-day total error|. Four baselines are always shown: mean of all history, of last 365 days, of last 90 days, and seasonal naive.

## 10. Models compared (36 regression models)

| Family | Models |
|---|---|
| Linear / regularised | Linear Regression, Ridge, Lasso, ElasticNet, Lasso-LARS, Orthogonal Matching Pursuit, Bayesian Ridge, ARD, PLS |
| Robust / GLM | Huber, Theil-Sen, RANSAC (failed on every fold), SGD, Passive-Aggressive, Poisson, Tweedie (power 1.5), Quantile (median) |
| Kernel / neighbours | KNN, SVR (RBF), NuSVR, Linear SVR, Kernel Ridge, Gaussian Process |
| Trees / ensembles | Decision Tree, Extra Tree, Random Forest, Extra Trees, Bagging, AdaBoost |
| Boosting | Gradient Boosting, HistGradient Boosting, XGBoost, LightGBM, CatBoost |
| Neural / voting | MLP, Voting (Ridge + Extra Trees + GBM + LGBM) |

`GammaRegressor` was left out because it needs strictly positive targets. This gives **70 successful model × feature-set combinations**. Scale-sensitive models run inside a `StandardScaler` pipeline.

### Results (validation = mean of 3 windows; lower is better)

| Rank | Model | Feature set | RMSE | WAPE 7d % | WAPE 30d % | 90-day total error % | Score |
|---|---|---|---|---|---|---|---|
| 1 | Linear Regression | A | 4,610 | 44.7 | 27.9 | 13.5 | 2.75 |
| 2 | Orthogonal Matching Pursuit | B | 4,625 | 45.1 | 30.3 | 14.4 | 5.25 |
| 3 | Lasso | A | 4,631 | 45.7 | 30.5 | 13.5 | 5.75 |
| 5 | Ridge | A | 4,631 | 45.2 | 29.1 | 14.4 | 5.75 |
| 10 | Tweedie GLM | B | 4,654 | 47.9 | 21.6 | 9.1 | 9.50 |
| – | Random Forest (best tree) | B | 4,680 | – | 35.7 | 15.1 | 20.0 |
| – | MLP neural net | A | 4,675 | – | 36.4 | 21.8 | 21.5 |
| – | XGBoost | A | 4,754 | – | 36.0 | 26.0 | 34.0 |
| – | Gradient Boosting | A | 5,321 | – | 74.0 | 72.4 | 64.0 |
| Baseline | mean of last 90 days | – | 4,642 | 50.0 | 33.4 | 19.5 | – |
| Baseline | mean of all history | – | 4,770 | 54.2 | 49.7 | 49.1 | – |
| Baseline | seasonal naive | – | 5,795 | 78.1 | 55.5 | 34.2 | – |

*The baselines' 90-day total error is the average absolute error over the three validation windows (same rule as for the models).*

**What the comparison shows**

1. **Differences are tiny.** The best model beats the best benchmark by only about 0.7 % in daily RMSE, less than the variation between windows. Every daily R² is about 0 or slightly negative.
2. **Simple regularised linear models on calendar features win.** Flexible models (forests, boosting, neural net, Gaussian process) fit the random spikes and generalise worse.
3. **Lag features did not help** on average (mean RMSE 4,783 for Set B vs 4,775 for Set A); recursive 90-day forecasting lets small errors accumulate.
4. The seasonal-naive baseline is the worst, since it copies last year's random spikes.

## 11. Hyper-parameter tuning (grid search on validation windows)

| Model | Best parameters | RMSE before → after | WAPE-30d before → after | 90-day total error before → after |
|---|---|---|---|---|
| Lasso | alpha = 0.5 | 4,631 → **4,607** | 30.5 → 27.4 | 13.5 → 14.5 |
| Ridge | alpha = 0.1 | 4,632 → 4,608 | 29.3 → 28.0 | 14.4 → 14.6 |
| Bayesian Ridge | alpha_2 = 1e-6, lambda_2 = 1 | 4,634 → 4,634 | no change | no change |

Linear Regression and Orthogonal Matching Pursuit have no tunable hyper-parameter, so they entered the final comparison unchanged. The best alphas sit at the lower edge of the grids, meaning the data wants very little shrinkage (Lasso at 0.5 is close to plain linear regression).

## 12. Final model selection

| Candidate | Features | Val. RMSE | WAPE-30d | Total error | RMSE std across windows | 90-day forecast time | Complexity | Score |
|---|---|---|---|---|---|---|---|---|
| **Lasso (alpha 0.5)** | A (20) | 4,607 | 27.4 | 14.5 | 2,276 | 8 ms | 20 coefficients | **1.75** |
| Linear Regression | A (20) | 4,610 | 27.9 | 13.5 | 2,254 | 8 ms | 20 coefficients | 2.25 |
| Ridge (alpha 0.1) | A (16) | 4,608 | 28.0 | 14.6 | 2,267 | 8 ms | 16 coefficients | 2.50 |
| Orthogonal Matching Pursuit | B (31) | 4,625 | 30.3 | 14.4 | 2,285 | 630 ms | 31 coefficients | 4.00 |
| Bayesian Ridge | A (16) | 4,634 | 29.5 | 14.9 | 2,295 | 8 ms | 16 coefficients | 4.50 |

**Why Lasso was deployed:** lowest average rank on the four validation metrics; calendar-only features mean it needs no history at serving time and forecasts 90 days in a single step (no error accumulation); 8 ms inference; a linear model whose 20 coefficients are easy to explain; the file is 1.5 KB. Lasso, Linear Regression and Ridge are practically tied, so this choice is about reliability and simplicity, not a big accuracy gap. The three most important features are seasonal: `month_sin` 39 %, `year_sin` 26 %, `month_cos` 13 % (together about 97 % with the other seasonal terms).

## 13. Test-window result (honest verdict)

Error of the 90-day **total** (forecast vs actual, %):

| Window | Role | Actual total | Selected model | Mean of last 90 days | Mean of all history |
|---|---|---|---|---|---|
| 05 Oct 2013 – 02 Jan 2014 | validation | 247,099 | −34.8 | −21.7 | −62.7 |
| 03 Jan 2014 – 02 Apr 2014 | validation | 203,766 | +0.1 | +21.3 | −48.5 |
| 03 Apr 2014 – 01 Jul 2014 | validation | 176,139 | +8.5 | +15.7 | −36.1 |
| 02 Jul 2014 – 29 Sep 2014 | **TEST** | 145,571 | **+71.4** | +21.0 | −19.5 |

- On the three validation windows the model's average absolute total error is **14.5 % vs 19.5 %** for the best benchmark.
- On the **test window** the selected model forecast about 249,000 against 145,571 actual (**+71 %**); its daily RMSE (4,412) and R² (−0.08) were worse than the "mean of last 90 days" benchmark (4,266). That window was **30 % below the same weeks of 2013**, while Jan–Sep 2014 overall was **+7 %** vs 2013. A model that extrapolates the growth trend is exposed to exactly this kind of slump, and no feature in this data could have predicted it.
- **Conclusion:** the model captures seasonal shape and growth, but a single quarter can deviate strongly. This is why the dashboard always shows the likely range and why the model should be re-trained when new data arrives.

## 14. Final forecast (30 Sep – 28 Dec 2014)

| Item | Value |
|---|---|
| Total, 90 days | **213,215** (likely range 122,592 – 286,264) |
| Average per day | 2,369 |
| By month | Oct 74,799 · Nov 53,110 · Dec 83,899 (plus 30 Sep: 1,406) |
| Same 90 days in 2013 (actual) | 243,463, so the forecast is **−12 %** |
| Last 90 days (actual) | 145,571 |
| Reasonableness | year-to-date 2014 is +7 % vs 2013, so −12 % is a cautious level |

**Uncertainty** comes from the real out-of-sample errors (actual − forecast, 5th and 95th percentile) of four back-test windows: daily −2,907 / +7,012; weekly −17,880 / +15,385; 30 days −49,413 / +48,753; 90 days −90,623 / +73,049. The dashboard band uses the daily values; totals for periods other than 90 days are scaled by the square root of the length (approximate).

## 15. Deployment architecture

User → **Streamlit UI** (`frontend/app.py`) → **FastAPI** (`api/main.py`) → shared feature code (`forecast_core.py`) → **Lasso model** (`model.joblib`) → forecast.

| Component | Details |
|---|---|
| Model files | `api/artifacts/model.joblib`, `metadata.json` (version 1.0.0, trained_at, data_hash, metrics, features, error bands, importance), `history.csv`, `forecast.csv` |
| API endpoints | `/health`, `/model-info`, `/forecast?days=`, `/forecast/date?date=`, `/forecast/range?start=&end=`, `/history`, `/leaderboard`, interactive docs at `/docs` |
| Production requirements | input validation (422 JSON errors), error handling, `requirements.txt` with pinned versions, environment variables (`MODEL_DIR`, `MAX_HORIZON`, `ALLOWED_ORIGINS`, `LOG_LEVEL`, `API_URL`), model versioning, health endpoint, Dockerfile (non-root user, HEALTHCHECK), request-ID logging |
| Docker / Render | `api/Dockerfile`, `frontend/Dockerfile`, `render.yaml` (two web services); the UI reads the API address from `API_URL` |
| Streamlit UI | KPI cards (90-day total, average/day, vs last year, vs last 90 days); forecast chart with uncertainty band (daily / weekly / monthly view); weekly and monthly tables; single-date and period lookup; sales-pattern charts; model, accuracy and feature-importance page; plain-language help; CSV download |

The UI shows **forecast value, likely range and important features** in place of the "predicted class / probability" mentioned in the assignment, because this is a regression problem. The assignment mentions AWS; this project uses Render + Docker as requested (the same images run on AWS).

## 16. Limitations and next steps

- Daily sales are mostly random bursts, so single days cannot be predicted precisely; totals over weeks and months are the reliable output.
- Only 3.7 years of data, with a changing business level; the most recent window was a large miss. Re-train regularly.
- Only sales history and the calendar are used. External information (marketing, holidays by country, large customer orders, price changes) could improve accuracy.
- Next steps: forecast per country or category and add them up, try weekly forecasting with daily spreading, add scheduled re-training, and track forecast errors in production.

## 17. Quick viva answers

- **Why regression models for a time series?** Date features (trend, seasonality) turn forecasting into a regression on a continuous target; this also lets us compare every regression model.
- **Why not a random train/test split?** It leaks the future; a rolling time split mimics real use.
- **Why do classification metrics not apply?** The target is continuous, so RMSE, WAPE and total error are used.
- **Why did a simple linear model win?** Daily sales are noisy; flexible models over-fit the noise, while regularised linear models capture trend and seasonality.
- **Why not MAPE?** Actual sales are 0 on 60 % of days (division by zero).
- **How is leakage avoided?** Same-day Quantity, COS, GP, Discount are excluded; lag features are shifted by one day; scaling and capping are fitted on training data only.
- **Why is the test result worse than validation?** The test quarter was an unusual slump (30 % below the previous year); the report shows this openly.

## Appendix – files

`notebook/Sales_Forecasting.ipynb` (executed notebook), `api/` (backend), `frontend/` (Streamlit app), `docs/figures/*.png` (18 charts), `docs/model_leaderboard.csv` (all 70 combinations), `docs/backtest_all_folds.csv`, `docs/final_candidates.csv`, `docs/tuning_results.csv`, `docs/feature_selection_comparison.csv`, `README.md` (how to run and deploy).
