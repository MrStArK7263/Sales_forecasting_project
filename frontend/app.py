"""
Sales Forecast Dashboard (Streamlit front-end)
Talks to the FastAPI backend whose address is given by the API_URL environment variable.
"""
import os
import time
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
BLUE, RED, GREY, GREEN, ORANGE = "#2F5D9E", "#D1495B", "#9AA5B1", "#2E9E6B", "#EDA33A"

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")
st.markdown("""
<style>
 .block-container {padding-top: 1.6rem;}
 div[data-testid="stMetric"] {background:#F5F7FA; border:1px solid #E3E8EF; border-radius:12px; padding:14px 16px;}
 @media (prefers-color-scheme: dark) { div[data-testid="stMetric"] {background:#1E242C; border:1px solid #2C3540;} }
 .hint {font-size:0.92rem; opacity:.85;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------ API helpers
def api_get(path: str, params: dict | None = None, retries: int = 3):
    """GET with retries – a free Render service can need ~30-60 s to wake up."""
    last = None
    for i in range(retries):
        try:
            r = requests.get(f"{API_URL}{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            raise ValueError(f"{r.status_code}: {detail}")
        except requests.exceptions.RequestException as e:
            last = e; time.sleep(2 + 3 * i)
    raise ConnectionError(str(last))


@st.cache_data(ttl=600, show_spinner=False)
def load_forecast(days: int):
    return api_get("/forecast", {"days": days})


@st.cache_data(ttl=600, show_spinner=False)
def load_history():
    return api_get("/history")


@st.cache_data(ttl=600, show_spinner=False)
def load_info():
    return api_get("/model-info"), api_get("/leaderboard")


def money(x: float) -> str:
    return f"{x:,.0f}"


# ------------------------------------------------------------------------------------------ sidebar
st.sidebar.title("📈 Sales Forecast")
st.sidebar.caption("Daily sales forecast for the next 3 months")
days = st.sidebar.slider("Forecast length (days)", 7, 90, 90, step=1, help="How many days after the last recorded day to show.")
view = st.sidebar.radio("Chart detail", ["Daily", "Weekly totals", "Monthly totals"], index=0)
show_band = st.sidebar.checkbox("Show uncertainty band", value=True)
hist_days = st.sidebar.select_slider("History shown on chart", options=[30, 60, 90, 180, 365], value=180)
st.sidebar.divider()
st.sidebar.caption(f"Backend: `{API_URL}`")

try:
    with st.spinner("Contacting the forecast service (the first call can take up to a minute if it was asleep)…"):
        fcst = load_forecast(days); hist = load_history(); info, board = load_info()
except Exception as e:
    st.error("The forecast service is not reachable right now.")
    st.info(f"Tried: {API_URL}. Check that the API is running (locally `uvicorn main:app`) or that `API_URL` points to the Render service.\n\nDetails: {e}")
    st.stop()

H = pd.DataFrame(hist["data"]); H["date"] = pd.to_datetime(H["date"]); H = H.set_index("date")["sales"]
F = pd.DataFrame(fcst["forecast"]); F["date"] = pd.to_datetime(F["date"]); F = F.set_index("date")
S = fcst["summary"]; last_actual = H.index.max()

# last year's same period (shifted by 364 days = same weekdays)
ly_idx = F.index - pd.Timedelta(days=364)
ly_total = float(H.reindex(ly_idx).fillna(0).sum()); recent_total = float(H.iloc[-days:].sum())

# ------------------------------------------------------------------------------------------ header + KPIs
st.title("Sales forecast – next 3 months")
st.markdown(f"Based on **{len(H):,} days** of history ({H.index.min():%d %b %Y} – {last_actual:%d %b %Y}). "
            f"Forecast covers **{F.index.min():%d %b %Y} → {F.index.max():%d %b %Y}**.")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Expected sales, next {days} days", money(S["total_forecast"]),
          help="Sum of the daily forecasts. The likely range is shown below the number.")
c1.caption(f"Likely range: {money(S['total_lower'])} – {money(S['total_upper'])}")
c2.metric("Average per day", money(S["average_per_day"]))
delta = (S["total_forecast"] / ly_total - 1) * 100 if ly_total > 0 else None
c3.metric("vs same period last year", money(ly_total), f"{delta:+.0f}%" if delta is not None else None,
          help="Actual sales in the same weekdays one year earlier (364 days before).")
delta2 = (S["total_forecast"] / recent_total - 1) * 100 if recent_total > 0 else None
c4.metric(f"vs last {days} days (actual)", money(recent_total), f"{delta2:+.0f}%" if delta2 is not None else None)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast", "🔎 Pick a date", "📊 Sales patterns", "🧠 Model & accuracy", "❓ How to read this"])

# ------------------------------------------------------------------------------------------ TAB 1 forecast
with tab1:
    st.subheader("Forecast vs recent history")
    Hh = H.iloc[-hist_days:]
    fig = go.Figure()
    if view == "Daily":
        fig.add_trace(go.Scatter(x=Hh.index, y=Hh.values, name="Actual daily sales", mode="lines", line=dict(color=GREY, width=1)))
        fig.add_trace(go.Scatter(x=Hh.index, y=Hh.rolling(14, min_periods=1).mean(), name="Actual – 14-day average", line=dict(color=BLUE, width=2.5)))
        if show_band:
            fig.add_trace(go.Scatter(x=list(F.index) + list(F.index[::-1]), y=list(F.upper) + list(F.lower[::-1]), fill="toself",
                                     fillcolor="rgba(209,73,91,0.13)", line=dict(width=0), name="Likely range (5–95 %)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=F.index, y=F.forecast, name="Forecast (expected sales)", line=dict(color=RED, width=3)))
        ymax = max(Hh.max(), F.forecast.max()) * 1.05
        fig.update_yaxes(range=[0, ymax])
    else:
        rule = "W-SUN" if view.startswith("Weekly") else "MS"
        hh = Hh.resample(rule).sum(); ff = F.forecast.resample(rule).sum()
        if rule == "MS":                                    # drop an incomplete first/last month so bars are comparable
            hh = hh.iloc[1:] if len(hh) > 1 else hh
        fig.add_trace(go.Bar(x=hh.index, y=hh.values, name="Actual", marker_color=BLUE))
        fig.add_trace(go.Bar(x=ff.index, y=ff.values, name="Forecast", marker_color=RED))
    fig.add_vline(x=last_actual.to_pydatetime(), line_dash="dot", line_color="#888")
    fig.add_annotation(x=last_actual, y=1.04, yref="paper", text="today (last data) ▸ forecast", showarrow=False, xanchor="left", font=dict(size=11))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15), hovermode="x unified",
                      yaxis_title="Sales", barmode="group")
    st.plotly_chart(fig, width="stretch")
    st.markdown("<div class='hint'>💡 Actual sales are very <b>spiky</b> (many days have no orders, a few days have huge orders). "
                "The red line is the <b>expected</b> sales of a day – the typical level you should plan for – not a prediction of individual spikes. "
                "Totals over weeks and months are far more reliable than any single day.</div>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Weekly totals")
        wk = F.forecast.resample("W-SUN").sum()
        figw = go.Figure(go.Bar(x=[f"{d - timedelta(days=6):%d %b}" for d in wk.index], y=wk.values, marker_color=RED,
                                text=[money(v) for v in wk.values], textposition="outside"))
        figw.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Forecast sales", xaxis_title="Week starting")
        st.plotly_chart(figw, width="stretch")
    with colB:
        st.subheader("Month by month")
        m = F.groupby(F.index.to_period("M")).agg(forecast=("forecast", "sum"), days=("forecast", "size"))
        lm = [float(H.reindex(F.index[F.index.to_period("M") == p] - pd.Timedelta(days=364)).fillna(0).sum()) for p in m.index]
        tbl = pd.DataFrame({"Month": [p.strftime("%b %Y") for p in m.index], "Days in forecast": m.days.values,
                            "Forecast sales": [money(v) for v in m.forecast], "Same days last year": [money(v) for v in lm],
                            "Change": [f"{(a / b - 1) * 100:+.0f}%" if b > 0 else "–" for a, b in zip(m.forecast, lm)]})
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption("Partial months only count the days inside the forecast window.")

    with st.expander("📋 Daily forecast table / download"):
        out = F.reset_index().rename(columns={"date": "Date", "weekday": "Weekday", "forecast": "Forecast", "lower": "Lower (5%)", "upper": "Upper (95%)"})
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(out, hide_index=True, width="stretch", height=300)
        st.download_button("⬇️ Download forecast as CSV", out.to_csv(index=False).encode(), "sales_forecast.csv", "text/csv")

# ------------------------------------------------------------------------------------------ TAB 2 date lookup
with tab2:
    st.subheader("Look up one day or a period")
    lo, hi = F.index.min().date(), F.index.max().date()
    d1, d2 = st.columns(2)
    with d1:
        pick = st.date_input("Choose a date", value=lo + timedelta(days=14), min_value=lo, max_value=hi)
        if st.button("Get forecast for this day", type="primary"):
            try:
                r = api_get("/forecast/date", {"date": str(pick)})
                st.metric(f"{r['weekday']} {pick:%d %b %Y}", money(r["forecast"]), help="Expected sales for this day")
                st.write(f"Likely range for a single day: **{money(r['lower'])} – {money(r['upper'])}**")
                st.info("A single day can be 0 (no orders) or far above the forecast. Use the period totals on the right for planning.")
            except Exception as e:
                st.error(f"Could not get the forecast: {e}")
    with d2:
        rng = st.date_input("…or choose a period", value=(lo, min(hi, lo + timedelta(days=29))), min_value=lo, max_value=hi)
        if isinstance(rng, tuple) and len(rng) == 2:
            try:
                r = api_get("/forecast/range", {"start": str(rng[0]), "end": str(rng[1])})
                s = r["summary"]
                st.metric(f"Expected sales {rng[0]:%d %b} – {rng[1]:%d %b}", money(s["total_forecast"]))
                st.write(f"Likely range: **{money(s['total_lower'])} – {money(s['total_upper'])}** · average per day **{money(s['average_per_day'])}**")
                st.caption("Range for periods other than 90 days is approximate.")
            except Exception as e:
                st.error(f"Could not get the forecast: {e}")

# ------------------------------------------------------------------------------------------ TAB 3 patterns
with tab3:
    st.subheader("What the history tells us")
    d = H.to_frame("sales"); d["year"] = d.index.year; d["month"] = d.index.month
    c1, c2 = st.columns(2)
    with c1:
        pv = d.pivot_table(index="month", columns="year", values="sales", aggfunc="sum")
        fm = go.Figure()
        for y in pv.columns:
            fm.add_trace(go.Scatter(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], y=pv[y].values, name=str(y), mode="lines+markers"))
        fm.update_layout(title="Monthly sales by year (yearly pattern)", height=360, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Sales")
        st.plotly_chart(fm, width="stretch")
        st.caption("2011 and 2014 are incomplete years. Months that peak in several years show the yearly seasonality the model learns.")
    with c2:
        dw = d.assign(dow=d.index.dayofweek).groupby("dow").sales.agg(avg="mean", share=lambda s: (s > 0).mean() * 100)
        fd = make_subplots(specs=[[{"secondary_y": True}]])
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fd.add_trace(go.Bar(x=names, y=dw["avg"], name="Average sales", marker_color=BLUE), secondary_y=False)
        fd.add_trace(go.Scatter(x=names, y=dw["share"], name="% days with orders", mode="lines+markers", line=dict(color=ORANGE)), secondary_y=True)
        fd.update_layout(title="Weekday pattern", height=360, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.2))
        fd.update_yaxes(title_text="Average daily sales", secondary_y=False); fd.update_yaxes(title_text="% of days with orders", secondary_y=True)
        st.plotly_chart(fd, width="stretch")
        st.caption("Weekdays differ only slightly – the day of the week is a weak signal compared with the time of year.")
    tr = go.Figure()
    tr.add_trace(go.Scatter(x=H.index, y=H.rolling(90, min_periods=30).sum(), name="Rolling 90-day sales", line=dict(color=BLUE, width=2.5)))
    tr.update_layout(title="Business trend: sales in the previous 90 days, over time", height=330, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Sales (90-day sum)")
    st.plotly_chart(tr, width="stretch")
    z = (H == 0).mean() * 100
    st.info(f"📌 **{z:.0f}% of all days had no orders at all.** This is why a single day cannot be predicted precisely, and why the dashboard emphasises weekly/monthly totals.")

# ------------------------------------------------------------------------------------------ TAB 4 model
with tab4:
    st.subheader("Which model is behind the forecast?")
    vm, tm, bb = info["validation_metrics"], info["test_metrics"], info["best_baseline"]
    st.success(f"**{info['model_name']}**  ·  version {info['model_version']}  ·  features: {info['feature_set']}  ·  chosen from "
               f"**{info['n_models_compared']} regression models** that were compared with time-based back-testing.")
    a, b, c = st.columns(3)
    a.metric("Error on weekly totals (WAPE, validation)", f"{vm['WAPE_7d']:.0f}%", help="Average |forecast − actual| of weekly totals ÷ actual total.")
    b.metric("Error on 30-day totals (WAPE, validation)", f"{vm['WAPE_30d']:.0f}%")
    c.metric("Error on 90-day total (last unseen window)", f"{tm['Total90_err']:+.0f}%", help="Forecast made 90 days ago vs what actually happened (never used for training/selection).")
    st.markdown(f"**Compared with a simple benchmark** (*{bb['model'].replace('Baseline: ', '')}*): the benchmark's 30-day-total error was "
                f"{bb['WAPE_30d']:.0f}%, the selected model's {vm['WAPE_30d']:.0f}%.")

    we = pd.DataFrame(info.get("window_errors", []))
    if not we.empty:
        st.markdown("#### How wrong was the 90-day total in each past test window?")
        bl_col = "BL: mean of last 90 days"
        fw = go.Figure()
        labels = [f"{w}<br>({r})" for w, r in zip(we["window"], we["role"])]
        fw.add_trace(go.Bar(x=labels, y=we["Selected model"], name="Selected model", marker_color=RED))
        if bl_col in we:
            fw.add_trace(go.Bar(x=labels, y=we[bl_col], name="Simple benchmark: average of last 90 days", marker_color=GREY))
        fw.add_hline(y=0, line_color="#555")
        fw.update_layout(barmode="group", height=330, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Forecast total vs actual total (%)",
                         legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fw, width="stretch")
        st.warning("⚠️ The most recent window (TEST) was an unusually weak quarter, and the model **over-forecast** it because it extrapolates the long-term "
                   "growth trend. Treat the forecast as a baseline planning scenario, look at the likely range rather than a single number, "
                   "and re-train the model when new data arrives.")

    col1, col2 = st.columns(2)
    with col1:
        lb = pd.DataFrame(board["top"]).head(10).iloc[::-1]
        fl = go.Figure(go.Bar(y=[f"{m} ({fs[0]})" for m, fs in zip(lb.model, lb.feature_set)], x=lb.score, orientation="h", marker_color=BLUE))
        fl.update_layout(title="Top-10 model combinations (average rank score, lower = better)", height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fl, width="stretch")
        st.caption("(A) = calendar features, (B) = calendar + recent-sales features.")
    with col2:
        fi = pd.Series(info["feature_importance"]).head(10).iloc[::-1]
        ff = go.Figure(go.Bar(y=fi.index, x=fi.values, orientation="h", marker_color=GREEN))
        ff.update_layout(title="What the model pays attention to (% importance)", height=400, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(ff, width="stretch")
    iv = info["intervals"]
    st.markdown(f"**How wide is the uncertainty?** In four past back-test windows, the 90-day total was off by between "
                f"**{money(iv['90day']['q05'])}** and **+{money(iv['90day']['q95'])}** (5–95 % of cases, actual − forecast). The shaded band on the chart uses these real past errors.")
    with st.expander("Technical details"):
        st.json({k: info[k] for k in ["model_name", "model_params", "features", "n_training_days", "history_start", "history_end", "trained_at", "data_hash"]})

# ------------------------------------------------------------------------------------------ TAB 5 help
with tab5:
    st.subheader("How to read this dashboard")
    st.markdown("""
**What is being forecast?** The *total sales of the whole business per day* for the 90 days after the last recorded order.

**Red line** – the *expected* sales for each day, learned from past years (growth trend + time of year + weekday) .
**Shaded band** – the range in which the actual value fell in 90 % of past back-tests. Individual days often land outside it
because many days have exactly 0 orders and a few have very large orders.

**Why not an exact daily number?** Daily sales here are random bursts. No model can know *which* day a big customer will order,
but it can estimate the typical level, the yearly pattern and the growth – which is what the weekly/monthly totals and the 90-day total show.

**How was it validated?** The model was always tested on data it had *never seen* (a later time window), repeated on four windows, and compared with
simple benchmarks, so the accuracy numbers are honest estimates of future performance.

**Good ways to use it:** plan stock and staffing from the weekly/monthly totals; treat the 90-day total and its range as the quarter's planning scenario;
re-train the model when new data arrives (see README).
""")
    st.caption(f"Model v{info['model_version']} · trained {info['trained_at']} · API: {API_URL}")
