"""Analyze the AMD data saved by collect_amd.py.

Run from the project root (after collect_amd.py):
    python src/analyze_amd.py

Needs:   data/AMD_bs_inputs.csv, data/AMD_options_history.csv
Writes:  figures/AMD_*.png, report/amd_summary.csv, report/amd_greeks_check.csv
Only depends on blackscholes.py (no other project files).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from blackscholes import bs_greeks, bs_price

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, REPORT = ROOT / "data", ROOT / "figures", ROOT / "report"
TD = 252


# ----------------------------------------------------------------------------
# A. Stock: does AMD look like geometric Brownian motion?
# ----------------------------------------------------------------------------
def stock_analysis(df):
    r = df["log_return"].dropna()
    n, T = len(r), len(r) / TD
    sigma = r.std(ddof=1) * np.sqrt(TD)
    mu = r.mean() * TD + 0.5 * sigma**2
    out = {
        "n_days": n, "start": df.index[0].date(), "end": df.index[-1].date(),
        "sigma_hat": sigma, "se_sigma": sigma / np.sqrt(2 * (n - 1)),
        "mu_hat": mu, "se_mu": sigma / np.sqrt(T),
        "skew": stats.skew(r), "excess_kurtosis": stats.kurtosis(r),
        "jarque_bera_p": stats.jarque_bera(r).pvalue,
        "shapiro_p": stats.shapiro(r).pvalue,
        # share of days beyond 3 sigma: normal predicts about 0.27%
        "pct_days_beyond_3sigma": float(np.mean(np.abs(r - r.mean()) > 3 * r.std(ddof=1)) * 100),
        # volatility clustering: autocorrelation of |returns| (zero if returns were iid)
        "abs_return_autocorr_lag1": float(pd.Series(np.abs(r.values)).autocorr(1)),
    }

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.hist(r, bins=60, density=True, alpha=0.6, label="Observed")
    x = np.linspace(r.min(), r.max(), 400)
    a1.plot(x, stats.norm.pdf(x, r.mean(), r.std(ddof=1)), "r", label="Fitted normal")
    a1.set(title="AMD daily log returns", xlabel="Log return"); a1.legend()
    stats.probplot(r, dist="norm", plot=a2); a2.set_title("Normal QQ plot")
    fig.tight_layout(); fig.savefig(FIG / "AMD_returns.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    for col, lab in [("vol_20d", "20-day"), ("vol_60d", "60-day"), ("vol_252d", "252-day")]:
        ax.plot(df.index, df[col], label=lab)
    ax.axhline(sigma, color="k", ls="--", lw=1, label=f"Constant sigma = {sigma:.2f}")
    ax.set(title="AMD rolling annualized volatility (GBM assumes it is constant)",
           ylabel="Volatility"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "AMD_rolling_vol.png", dpi=150); plt.close(fig)
    return out


# ----------------------------------------------------------------------------
# B. Options: smile and model-vs-market pricing error (latest snapshot)
# ----------------------------------------------------------------------------
def option_analysis(opt):
    day = opt["snapshot_date"].max()
    s = opt[opt["snapshot_date"] == day].copy()
    s = s[s["implied_vol_bs"].between(0.05, 3.0)]
    s["error"] = s["bs_price_hist_vol"] - s["mid"]
    s["rel_error"] = s["error"] / s["mid"]
    bins = [0, 0.90, 0.97, 1.03, 1.10, 10]
    labels = ["deep ITM (K<0.90S)", "ITM (0.90-0.97)", "ATM (0.97-1.03)",
              "OTM (1.03-1.10)", "far OTM (K>1.10S)"]
    s["bucket"] = pd.cut(s["moneyness_K_over_S"], bins=bins, labels=labels)
    table = s.groupby("bucket", observed=True).agg(
        n=("mid", "size"), mean_implied_vol=("implied_vol_bs", "mean"),
        mean_error=("error", "mean"), mean_rel_error=("rel_error", "mean"))

    fig, ax = plt.subplots(figsize=(8, 4))
    for exp, g in s.groupby("expiry"):
        g = g.sort_values("moneyness_K_over_S")
        ax.plot(g["moneyness_K_over_S"], g["implied_vol_bs"], "o-", ms=3, label=exp)
    ax.axhline(s["sigma_hist"].iloc[0], color="k", ls="--", lw=1,
               label="Historical sigma (60d)")
    ax.set(title=f"AMD implied volatility smile ({day})", xlabel="Strike / spot",
           ylabel="Implied volatility")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "AMD_smile.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(s["mid"], s["bs_price_hist_vol"], s=12, alpha=0.7)
    lim = [0, max(s["mid"].max(), s["bs_price_hist_vol"].max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="Perfect agreement")
    ax.set(title="Market mid vs Black-Scholes (historical vol)", xlabel="Market mid price",
           ylabel="Model price"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "AMD_model_vs_market.png", dpi=150); plt.close(fig)

    summary = {
        "snapshot_date": day, "n_quotes_used": len(s),
        "sigma_hist_60d": float(s["sigma_hist"].iloc[0]),
        "mean_implied_vol": float(s["implied_vol_bs"].mean()),
        "atm_implied_vol": float(s.loc[(s["moneyness_K_over_S"] - 1).abs().idxmin(), "implied_vol_bs"]),
        "rmse_price": float(np.sqrt((s["error"] ** 2).mean())),
        "mean_rel_error": float(s["rel_error"].mean()),
    }
    return summary, table, s


# ----------------------------------------------------------------------------
# C. Greeks: analytic formulas vs finite differences (your proposal's "known values")
# ----------------------------------------------------------------------------
def greeks_check(s):
    pick = s[s["days_to_expiry"].between(30, 120)]
    row = pick.loc[(pick["moneyness_K_over_S"] - 1).abs().idxmin()] if len(pick) else \
        s.loc[(s["moneyness_K_over_S"] - 1).abs().idxmin()]
    S, K, T, r = float(row["spot"]), float(row["strike"]), float(row["T_years"]), float(row["rf_cc"])
    sig = float(row["implied_vol_bs"])
    f = lambda S=S, T=T, r=r, sg=sig: bs_price(S, K, T, r, sg, "call")
    hS, hs, hT, hr = S * 1e-4, 1e-4, 1e-5, 1e-6
    fd = {
        "delta": (f(S=S + hS) - f(S=S - hS)) / (2 * hS),
        "gamma": (f(S=S + hS) - 2 * f() + f(S=S - hS)) / hS**2,
        "vega": (f(sg=sig + hs) - f(sg=sig - hs)) / (2 * hs),
        "theta": -(f(T=T + hT) - f(T=T - hT)) / (2 * hT),   # theta = -dC/dT (per year)
        "rho": (f(r=r + hr) - f(r=r - hr)) / (2 * hr),
    }
    an = bs_greeks(S, K, T, r, sig, "call")
    out = pd.DataFrame({"analytic": {k: float(v) for k, v in an.items()},
                        "finite_difference": fd})
    out["abs_diff"] = (out["analytic"] - out["finite_difference"]).abs()
    out.attrs["contract"] = f"S={S:.2f}, K={K:.0f}, T={T:.3f}y, r={r:.4f}, sigma={sig:.3f}"
    return out


# ----------------------------------------------------------------------------
def main():
    FIG.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    df = pd.read_csv(DATA / "AMD_bs_inputs.csv", index_col=0, parse_dates=True)
    pd.set_option("display.width", 200, "display.max_columns", None)

    st = stock_analysis(df)
    print("=== A. STOCK: GBM fit and assumption checks ===")
    for k, v in st.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")

    opt_path = DATA / "AMD_options_history.csv"
    if not opt_path.exists():
        print("\nNo AMD_options_history.csv yet. Run collect_amd.py during market hours.")
        pd.Series(st).to_csv(REPORT / "amd_summary.csv", header=["value"])
        return
    opt = pd.read_csv(opt_path)
    summ, table, used = option_analysis(opt)
    print("\n=== B. OPTIONS: model vs market (latest snapshot) ===")
    for k, v in summ.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")
    print("\n  Pricing error by moneyness (model - market, historical vol):")
    print(table.round(4).to_string())

    gk = greeks_check(used)
    print(f"\n=== C. GREEKS check on {gk.attrs['contract']} ===")
    print(gk.to_string(float_format=lambda v: f"{v:.6g}"))

    pd.Series({**st, **summ}).to_csv(REPORT / "amd_summary.csv", header=["value"])
    table.to_csv(REPORT / "amd_error_by_moneyness.csv")
    gk.to_csv(REPORT / "amd_greeks_check.csv")
    print(f"\nSaved figures to {FIG} and tables to {REPORT}")


if __name__ == "__main__":
    main()