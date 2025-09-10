import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import asdict, dataclass
from typing import Literal, Tuple, Optional

from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="CPI Risk Forecaster", layout="wide")

# ---------------------------
# Utilities
# ---------------------------

def dense_rank_desc(values: pd.Series) -> pd.Series:
    # Dense rank: 1,2,3,... without gaps, higher score = better rank (1 is best)
    # We want higher score => lower rank number. So sort descending.
    order = values.rank(method="dense", ascending=False).astype(int)
    return order

def average_rank_desc(values: pd.Series) -> pd.Series:
    # Average rank (like competition average), higher score = better rank
    order = values.rank(method="average", ascending=False)
    return order

def reconstruct_ranks(df: pd.DataFrame, tie_rule: Literal["dense","average"]="dense") -> pd.DataFrame:
    assert {"country","year","score"}.issubset(df.columns)
    rank_fn = dense_rank_desc if tie_rule=="dense" else average_rank_desc
    out = []
    for y, g in df.groupby("year", as_index=False):
        r = rank_fn(g["score"])
        gg = g.copy()
        gg["rank"] = r
        gg["n_countries"] = len(g)
        out.append(gg)
    return pd.concat(out, ignore_index=True)

def fit_arima(y: pd.Series, order: Tuple[int,int,int]) -> ARIMA:
    # y must be indexed by year increasing
    model = ARIMA(y, order=order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit()
    return res

def simulate_paths_arima(res, horizon: int, n_sims: int, seed: int = 42) -> np.ndarray:
    """
    Try to use statsmodels' simulate for dynamic Monte Carlo paths.
    Fallback: draw from forecast mean + normal noise per step (approx).
    Returns array shape (n_sims, horizon).
    """
    rng = np.random.default_rng(seed)
    try:
        sims = res.simulate(nsimulations=horizon, repetitions=n_sims, anchor="end", random_errors=rng)
        # simulate returns (horizon, n_sims) -> transpose
        return np.asarray(sims).T
    except Exception:
        # Fallback: independent normals with step-wise std from get_forecast
        fc = res.get_forecast(steps=horizon)
        mean = fc.predicted_mean.values  # (H,)
        se = np.sqrt(np.diag(fc.covariance_matrix))  # (H,)
        draws = rng.normal(loc=mean, scale=se, size=(n_sims, horizon))
        return draws

def apply_shock_schedule(paths: np.ndarray, shock_first_year: float, shock_linger: float) -> np.ndarray:
    """
    paths: (n_sims, H)
    shock_first_year: additive delta to step 1 (e.g., -3.0)
    shock_linger: additive delta to steps 2..H (e.g., -1.0)
    """
    paths_adj = paths.copy()
    if paths_adj.shape[1] >= 1:
        paths_adj[:, 0] += shock_first_year
    if paths_adj.shape[1] >= 2:
        paths_adj[:, 1:] += shock_linger
    return paths_adj

def scores_to_ranks(scores: np.ndarray, ref_scores: np.ndarray, tie_rule: Literal["dense","average"]="dense") -> np.ndarray:
    """
    Map simulated scores to ranks using the reference-year global score distribution.
    Higher score => better rank (1 best). We compute where each simulated score would sit.
    """
    # Sort ref scores descending
    ref = np.sort(ref_scores)[::-1]
    # For each score s, count how many ref scores are strictly greater (or greater/equal depending on tie rule).
    # Dense rule: rank = 1 + number of unique ref scores strictly greater than s
    # Average rule: approximate by midpoint within ties using ECDF continuity correction.
    ranks = np.empty_like(scores, dtype=float)

    if tie_rule == "dense":
        unique_vals = np.unique(ref)
        # Precompute a mapping: value -> dense rank
        # Dense ranks by unique sorted descending
        dense_map = {val: i+1 for i, val in enumerate(unique_vals[::-1])}  # careful: unique sorted ascending -> reverse
        # Actually simpler: compute rank as 1 + count of unique ref values strictly greater
        unique_desc = np.sort(unique_vals)[::-1]
        def dense_rank_val(v):
            return 1 + np.sum(unique_desc > v)
        vec_dense = np.vectorize(dense_rank_val)
        ranks = vec_dense(scores)
    else:
        # Average rule: rank = 1 + (# strictly greater) + 0.5*(# equal - 1)
        def avg_rank_val(v):
            greater = np.sum(ref > v)
            equal = np.sum(ref == v)
            if equal == 0:
                # Find insertion point (descending)
                return greater + 1
            return greater + (equal + 1)/2.0
        vec_avg = np.vectorize(avg_rank_val)
        ranks = vec_avg(scores)

    return ranks

def current_rank_for_country(df_ranked: pd.DataFrame, country: str) -> Tuple[int, int, float, int]:
    """
    Returns (latest_year, current_rank, current_score, n_countries_that_year)
    """
    d = df_ranked[df_ranked["country"] == country]
    latest_year = d["year"].max()
    row = d[d["year"] == latest_year].iloc[0]
    return int(latest_year), int(round(row["rank"])), float(row["score"]), int(row["n_countries"])

@dataclass
class RunSpec:
    country: str
    arima_order: Tuple[int, int, int]
    horizon_year: int
    n_sims: int
    seed: int
    tie_rule: str
    rank_threshold_delta: int
    shock_first_year: float
    shock_linger: float
    reference_year: int

# ---------------------------
# Sidebar: Inputs
# ---------------------------

st.title("CPI Risk Forecaster (Simple)")

st.sidebar.header("1) Data")
uploaded = st.sidebar.file_uploader("Upload CSV (columns: country, year, score)", type=["csv"])

if uploaded is None:
    st.info(
        "Upload a CSV with columns **country, year, score**. "
        "Tip: Use Transparency International / Our World in Data CPI extracts."
    )
    demo = pd.DataFrame({
        "country": ["Mongolia"]*12 + ["Kazakhstan"]*12 + ["Kyrgyzstan"]*12,
        "year":    list(range(2013, 2025))*3,
        "score":   # demo numbers (not real)
                    [38,39,39,38,36,35,35,35,34,33,33,33] +  # Mongolia-ish
                    [29,29,31,31,31,34,34,35,36,36,37,37] +  # KZ
                    [24,24,25,25,26,26,28,28,27,28,28,29]     # KG
    })
    df = demo.copy()
else:
    df = pd.read_csv(uploaded)
    # Basic checks
    missing = {"country","year","score"} - set(df.columns)
    if missing:
        st.error(f"CSV missing required columns: {missing}")
        st.stop()


df["country"] = df["country"].astype(str)
df["year"] = df["year"].astype(int)
df["score"] = df["score"].astype(float)

st.sidebar.header("2) Ranking")
tie_rule = st.sidebar.radio("Tie rule for per-year ranks", ["dense","average"], index=0, help="Dense = 1,2,3 with no gaps. Average = competition average rank within ties.")

df_ranked = reconstruct_ranks(df, tie_rule=tie_rule)

countries = sorted(df["country"].unique().tolist())
country = st.sidebar.selectbox("Country", countries, index=min(0, len(countries)-1))

# Determine latest obs & reference-year options
latest_year_global = int(df["year"].max())
years = sorted(df["year"].unique().tolist())
ref_year = st.sidebar.selectbox(
    "Reference year for rank mapping (global distribution)",
    options=years,
    index=years.index(latest_year_global),
    help="Simulated scores are mapped to ranks using the global score distribution of this year."
)

st.sidebar.header("3) Model")
p = st.sidebar.number_input("ARIMA p (AR lags)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("ARIMA d (diffs)", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("ARIMA q (MA lags)", min_value=0, max_value=5, value=0, step=1)

st.sidebar.header("4) Simulation")
target_year = st.sidebar.number_input("Forecast to year", min_value=latest_year_global+1, max_value=2100, value=max(latest_year_global+6, latest_year_global+1), step=1)
n_sims = st.sidebar.number_input("# Monte Carlo draws", min_value=500, max_value=20000, value=5000, step=500)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=10**9, value=42, step=1)

st.sidebar.header("5) Shock (optional)")
shock_on = st.sidebar.checkbox("Apply scandal shock", value=False)
shock_first = st.sidebar.number_input("Δ score in first forecast year", value=-3.0, step=0.5, help="E.g., -3.0 for a sharp hit in the first forecast step.")
shock_linger = st.sidebar.number_input("Δ score in subsequent years", value=-1.0, step=0.5, help="E.g., -1.0 persistent penalty after the first year.")

st.sidebar.header("6) Risk metric")
k_drop = st.sidebar.number_input("K: rank worsens by ≥ K places", min_value=1, max_value=100, value=10, step=1)

# ---------------------------
# Main computation
# ---------------------------

st.subheader("Data Preview")
with st.expander("Show data (first 500 rows)"):
    st.dataframe(df_ranked.sort_values(["year","rank"]).head(500), use_container_width=True)

# Prepare time series for the chosen country
d_country = df_ranked[df_ranked["country"] == country].sort_values("year")
if d_country.empty:
    st.error("No data for the selected country.")
    st.stop()

y = d_country.set_index("year")["score"].astype(float)
years_country = y.index.values
last_year_country = int(years_country.max())

# Horizon length
H = int(target_year - last_year_country)
if H <= 0:
    st.warning(f"Target year must be > {last_year_country}.")
    st.stop()

# Fit ARIMA
try:
    res = fit_arima(y, (p,d,q))
except Exception as e:
    st.error(f"ARIMA failed to fit: {e}")
    st.stop()

# Deterministic forecast (for fan plot)
fc = res.get_forecast(steps=H)
fc_mean = pd.Series(fc.predicted_mean.values, index=np.arange(last_year_country+1, target_year+1))
ci = fc.conf_int(alpha=0.2)  # 80% CI
ci95 = fc.conf_int(alpha=0.05)  # 95% CI
ci.index = fc_mean.index
ci95.index = fc_mean.index

# Monte Carlo simulation
paths = simulate_paths_arima(res, horizon=H, n_sims=int(n_sims), seed=int(seed))
if shock_on:
    paths = apply_shock_schedule(paths, shock_first, shock_linger)

# Reference distribution for rank mapping
ref_dist = df_ranked[df_ranked["year"] == ref_year]["score"].dropna().values
if len(ref_dist) < 10:
    st.warning("Reference year has few countries; rank mapping may be unstable.")

# Convert each simulated 2030 (target_year) score to a rank
scores_2030 = paths[:, -1]
ranks_2030 = scores_to_ranks(scores_2030, ref_dist, tie_rule=tie_rule)

# Current rank (latest observed for the country)
cur_year, cur_rank, cur_score, cur_n = current_rank_for_country(df_ranked, country)

# Risk metric: P(rank >= cur_rank + K)
threshold_rank = cur_rank + int(k_drop)
prob_worse = float(np.mean(ranks_2030 >= threshold_rank))

# ---------------------------
# Display
# ---------------------------

cols = st.columns(4)
cols[0].metric("Country", country)
cols[1].metric("Latest observed year", f"{cur_year}")
cols[2].metric("Current rank", f"{cur_rank} / {cur_n}")
cols[3].metric(f"P(Δrank ≥ {k_drop}) by {target_year}", f"{prob_worse*100:.1f}%")

st.caption(f"Ranks reconstructed with **{tie_rule}** rule; simulated ranks mapped using **{ref_year}** global distribution.")

# Fan chart of scores
st.subheader("Score Forecast (fan)")
fig1, ax1 = plt.subplots(figsize=(7.5,4.5))
ax1.plot(y.index, y.values, label="Observed", linewidth=2)
ax1.plot(fc_mean.index, fc_mean.values, label="Point forecast", linewidth=2)
# 80% band
ax1.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3, label="80% CI")
# 95% band
ax1.fill_between(ci95.index, ci95.iloc[:,0], ci95.iloc[:,1], alpha=0.15, label="95% CI")
ax1.axvline(x=last_year_country, linestyle="--", alpha=0.5)
ax1.set_xlabel("Year"); ax1.set_ylabel("CPI score (0-100)")
ax1.legend(loc="best")
st.pyplot(fig1, use_container_width=True)

# Histogram of 2030 ranks
st.subheader(f"Simulated {target_year} Ranks")
fig2, ax2 = plt.subplots(figsize=(7.5,4.5))
ax2.hist(ranks_2030, bins=20)
ax2.axvline(cur_rank, linestyle="--", label=f"Current rank ({cur_rank})")
ax2.axvline(threshold_rank, linestyle="--", label=f"Threshold ({threshold_rank})")
ax2.set_xlabel("Rank (higher is worse)"); ax2.set_ylabel("Count of simulations")
ax2.legend(loc="best")
st.pyplot(fig2, use_container_width=True)

# ---------------------------
# Downloads
# ---------------------------

st.subheader("Download")
# Results table
out_df = pd.DataFrame({
    "sim_id": np.arange(len(scores_2030)),
    "score_target": scores_2030,
    "rank_target": ranks_2030
})
csv_buf = io.StringIO()
out_df.to_csv(csv_buf, index=False)

spec = RunSpec(
    country=country,
    arima_order=(int(p), int(d), int(q)),
    horizon_year=int(target_year),
    n_sims=int(n_sims),
    seed=int(seed),
    tie_rule=tie_rule,
    rank_threshold_delta=int(k_drop),
    shock_first_year=float(shock_first if shock_on else 0.0),
    shock_linger=float(shock_linger if shock_on else 0.0),
    reference_year=int(ref_year),
)
spec_json = json.dumps(asdict(spec), indent=2)

st.download_button("Download simulations CSV", data=csv_buf.getvalue(), file_name=f"cpi_mc_{country}_{target_year}.csv", mime="text/csv")
st.download_button("Download run spec (JSON)", data=spec_json, file_name=f"run_spec_{country}_{target_year}.json", mime="application/json")

# ---------------------------
# Notes / Help
# ---------------------------

with st.expander("How this works"):
    st.markdown("""
- **Ranks per year** are reconstructed from scores (higher score → better rank).
- **ARIMA(p,d,q)** is fit to the selected country's annual CPI score.
- We **simulate** many possible score paths to the target year.
- We **optionally apply a scandal shock**: a one-off hit in the first forecast year and a lingering penalty thereafter.
- Each simulated target-year score is converted to a **rank** using the **global score distribution** in your chosen reference year.
- We report **P(Δrank ≥ K)** where Δrank = (simulated rank − current rank).
    """)

with st.expander("Tips"):
    st.markdown("""
- Use **dense** ranking for simple, gapless ranks; **average** gives competition-style averages within ties.
- If you have a full panel from 1995+, your ARIMA will be more stable.
- For robustness, try a few **ARIMA orders** and compare probabilities.
- The **reference year** matters: choose a recent year with broad country coverage.
- To model reforms, use **positive shocks** (e.g., +2 then +0.5 lingering).
    """)
