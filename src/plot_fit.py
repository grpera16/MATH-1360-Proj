"""Plot fitted solutions of the Black-Scholes model.

Run from the project root:
    python src/plot_fit.py AMD            # uses data/AMD_bs_inputs.csv (+ options history)
    python src/plot_fit.py ASML_AS        # uses data/ASML_AS_prices.csv
    python src/plot_fit.py 0981_HK        # SMIC (Hong Kong)
    python src/plot_fit.py AMD --train 0.7

Figure 1 (<name>_fitted_gbm.png): the fitted stock-price solution
    S(t) = S0 exp((mu - sigma^2/2) t + sigma W(t))
fit on the first part of the data, then projected forward with its median, mean
and 50% / 90% bands, against what the price actually did.

Figure 2 (<name>_fitted_bs_curve.png), if option data exists: Black-Scholes call
price vs strike, with historical volatility and with the single best-fit volatility,
against market mid prices.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from blackscholes import bs_price

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
TD = 252


def load_prices(name):
    """Accept either collect_amd.py output or data.py output."""
    a, b = DATA / f"{name}_bs_inputs.csv", DATA / f"{name}_prices.csv"
    if a.exists():
        return pd.read_csv(a, index_col=0, parse_dates=True)["close"]
    if b.exists():
        return pd.read_csv(b, index_col=0, parse_dates=True)["Close"]
    raise FileNotFoundError(f"Neither {a.name} nor {b.name} found in data/. "
                            "Run collect_amd.py or analysis.py first.")


def fit_gbm(prices):
    r = np.log(prices / prices.shift(1)).dropna()
    sigma = r.std(ddof=1) * np.sqrt(TD)
    mu = r.mean() * TD + 0.5 * sigma**2
    return mu, sigma


def plot_gbm(name, prices, train_frac):
    k = int(len(prices) * train_frac)
    train, test = prices.iloc[:k], prices.iloc[k - 1:]
    mu, sigma = fit_gbm(train)
    S0, n = float(train.iloc[-1]), len(test) - 1
    t = np.arange(n + 1) / TD                      # years since the split date
    drift = (mu - 0.5 * sigma**2) * t              # drift of log price

    def q(p):                                      # exact GBM quantile at probability p
        return S0 * np.exp(drift + sigma * np.sqrt(t) * norm.ppf(p))

    median, mean = S0 * np.exp(drift), S0 * np.exp(mu * t)
    inside90 = float(np.mean((test.values >= q(0.05)) & (test.values <= q(0.95))) * 100)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train.values, color="black", lw=1.2, label="Actual (used to fit)")
    ax.plot(test.index, test.values, color="red", lw=1.6, label="Actual (held out)")
    rng = np.random.default_rng(1360)
    Z = rng.standard_normal((15, n))
    paths = S0 * np.exp(np.hstack([np.zeros((15, 1)),
                                   np.cumsum((mu - .5 * sigma**2) / TD
                                             + sigma / np.sqrt(TD) * Z, axis=1)]))
    for p in paths:
        ax.plot(test.index, p, color="grey", alpha=0.2, lw=0.7)
    ax.fill_between(test.index, q(0.05), q(0.95), color="tab:blue", alpha=0.15, label="90% band")
    ax.fill_between(test.index, q(0.25), q(0.75), color="tab:blue", alpha=0.25, label="50% band")
    ax.plot(test.index, median, color="tab:blue", lw=2, label="Fitted median")
    ax.plot(test.index, mean, color="tab:blue", lw=1.5, ls="--", label="Fitted mean")
    ax.axvline(train.index[-1], color="k", ls=":", lw=1)
    ax.set(title=f"{name}: fitted GBM solution (mu={mu:.2f}, sigma={sigma:.2f}) vs actual",
           ylabel="Price")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"{name}_fitted_gbm.png", dpi=150); plt.close(fig)
    print(f"GBM fit on first {train_frac:.0%} ({len(train)} days): mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"Held-out actual path inside the 90% band on {inside90:.1f}% of days")


def plot_bs_curve(name):
    path = DATA / f"{name}_options_history.csv"
    if not path.exists():
        print(f"(No {path.name}; skipping the option-price figure.)")
        return
    opt = pd.read_csv(path)
    opt = opt[opt["snapshot_date"] == opt["snapshot_date"].max()]
    exp = opt["expiry"].value_counts().idxmax()        # expiry with the most quotes
    g = opt[opt["expiry"] == exp].sort_values("strike")
    S, T, r = float(g["spot"].iloc[0]), float(g["T_years"].iloc[0]), float(g["rf_cc"].iloc[0])
    sig_hist = float(g["sigma_hist"].iloc[0])
    K, mid = g["strike"].values, g["mid"].values

    sse = lambda s: float(np.sum((bs_price(S, K, T, r, s, "call") - mid) ** 2))
    sig_fit = float(minimize_scalar(sse, bounds=(0.01, 3.0), method="bounded").x)

    ks = np.linspace(K.min(), K.max(), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(K, mid, color="black", s=18, zorder=3, label="Market mid price")
    ax.plot(ks, bs_price(S, ks, T, r, sig_hist, "call"), "tab:orange",
            label=f"Black-Scholes, historical sigma = {sig_hist:.2f}")
    ax.plot(ks, bs_price(S, ks, T, r, sig_fit, "call"), "tab:blue", lw=2,
            label=f"Black-Scholes, best-fit sigma = {sig_fit:.2f}")
    ax.axvline(S, color="k", ls=":", lw=1)
    ax.set(title=f"{name} calls expiring {exp}: Black-Scholes fitted to market",
           xlabel="Strike", ylabel="Call price")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"{name}_fitted_bs_curve.png", dpi=150); plt.close(fig)
    rm = lambda s: float(np.sqrt(np.mean((bs_price(S, K, T, r, s, "call") - mid) ** 2)))
    print(f"Option fit for {exp}: best-fit sigma={sig_fit:.4f} (RMSE {rm(sig_fit):.3f}) "
          f"vs historical sigma={sig_hist:.4f} (RMSE {rm(sig_hist):.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="AMD, ASML_AS, or 0981_HK")
    ap.add_argument("--train", type=float, default=0.8, help="fraction of data used to fit")
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    plot_gbm(a.name, load_prices(a.name), a.train)
    plot_bs_curve(a.name)