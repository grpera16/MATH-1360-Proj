"""Collect AMD data that feeds Black-Scholes, and save it to CSV.

Run from the project root:
    python src/collect_amd.py                 # stock history + today's option snapshot
    python src/collect_amd.py --no-options    # stock history only
    python src/collect_amd.py --years 10

Outputs (in data/):
    AMD_bs_inputs.csv          daily history: price, log return, rolling vol,
                               risk-free rate, dividends
    AMD_options_<date>.csv     today's call-option chain with market mid prices,
                               implied vol, and the Black-Scholes price
    AMD_options_history.csv    every snapshot appended together

Why snapshots? Yahoo only serves the CURRENT option chain, not past ones. Run this
script once per trading day to build your own option history.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from blackscholes import bs_price, implied_vol

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRADING_DAYS = 252


# ----------------------------------------------------------------------------
# Part 1: daily stock history and Black-Scholes inputs
# ----------------------------------------------------------------------------
def _clean_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df


def build_history(ticker="AMD", years=5):
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}")
    hist = _clean_index(hist)

    df = pd.DataFrame(index=hist.index)
    df["close"] = hist["Close"]                       # split-adjusted close
    df["volume"] = hist["Volume"]
    df["dividends"] = hist.get("Dividends", 0.0)       # should be all zeros for AMD
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling annualized volatility = candidate sigma inputs for Black-Scholes
    for w in (20, 60, 252):
        df[f"vol_{w}d"] = df["log_return"].rolling(w).std(ddof=1) * np.sqrt(TRADING_DAYS)

    # Risk-free rate: 13-week T-bill yield (^IRX, quoted in percent)
    try:
        irx = _clean_index(yf.Ticker("^IRX").history(period=f"{years}y"))["Close"]
        rf = (irx / 100.0).reindex(df.index).ffill().bfill()
    except Exception as e:
        print(f"!! Could not download ^IRX ({e}); leaving rf blank. Fill it in manually.")
        rf = pd.Series(np.nan, index=df.index)
    df["rf_3m"] = rf                                   # simple annual rate, decimal
    df["rf_3m_cc"] = np.log1p(rf)                      # continuously compounded (what BS uses)

    return df


# ----------------------------------------------------------------------------
# Part 2: today's option chain (calls only: no early-exercise issue for a
# non-dividend stock, so American calls = European calls)
# ----------------------------------------------------------------------------
def snapshot_calls(ticker, spot, rf_cc, sigma_hist, min_days=7, max_days=400,
                   moneyness_band=0.30):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    today = pd.Timestamp.today().normalize()
    rows = []
    for exp in tk.options:
        days = (pd.Timestamp(exp) - today).days
        if days < min_days or days > max_days:
            continue
        calls = tk.option_chain(exp).calls
        T = days / 365.0
        for _, c in calls.iterrows():
            K = float(c["strike"])
            if abs(K / spot - 1) > moneyness_band:
                continue
            bid, ask = float(c.get("bid", 0) or 0), float(c.get("ask", 0) or 0)
            if bid <= 0 or ask <= 0 or ask < bid:
                continue                                # stale or missing quote
            mid = 0.5 * (bid + ask)
            intrinsic = max(spot - K * np.exp(-rf_cc * T), 0.0)
            try:
                iv = implied_vol(mid, spot, K, T, rf_cc, "call") if mid > intrinsic else np.nan
            except ValueError:
                iv = np.nan
            rows.append({
                "snapshot_date": today.date(), "expiry": exp, "days_to_expiry": days,
                "T_years": T, "strike": K, "spot": spot,
                "moneyness_K_over_S": K / spot,
                "bid": bid, "ask": ask, "mid": mid, "last": c.get("lastPrice"),
                "volume": c.get("volume"), "open_interest": c.get("openInterest"),
                "yahoo_iv": c.get("impliedVolatility"),
                "implied_vol_bs": iv,
                "rf_cc": rf_cc, "sigma_hist": sigma_hist,
                "bs_price_hist_vol": bs_price(spot, K, T, rf_cc, sigma_hist, "call"),
            })
    return pd.DataFrame(rows)


def append_history(snapshot, path):
    snapshot = snapshot.copy()
    snapshot["snapshot_date"] = snapshot["snapshot_date"].astype(str)
    if path.exists():
        old = pd.read_csv(path)
        old["snapshot_date"] = old["snapshot_date"].astype(str)
        snapshot = pd.concat([old, snapshot], ignore_index=True)
    key = ["snapshot_date", "expiry", "strike"]
    snapshot = snapshot.drop_duplicates(subset=key, keep="last")
    snapshot.to_csv(path, index=False)
    return len(snapshot)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AMD")
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--no-options", action="store_true")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    df = build_history(args.ticker, args.years)
    out = DATA_DIR / f"{args.ticker}_bs_inputs.csv"
    df.to_csv(out)

    spot = float(df["close"].iloc[-1])
    sigma = float(df["vol_60d"].dropna().iloc[-1])
    rf_cc = float(df["rf_3m_cc"].dropna().iloc[-1]) if df["rf_3m_cc"].notna().any() else 0.04
    print(f"Saved {len(df)} rows to {out}")
    print(f"  {df.index[0].date()} -> {df.index[-1].date()}   spot={spot:.2f}")
    print(f"  60-day vol={sigma:.3f}   252-day vol={df['vol_252d'].iloc[-1]:.3f}   rf(cc)={rf_cc:.4f}")
    print(f"  Total dividends in period: {df['dividends'].sum():.4f}  (0 means the no-dividend assumption holds)")

    if args.no_options:
        return
    snap = snapshot_calls(args.ticker, spot, rf_cc, sigma)
    if snap.empty:
        print("No usable option quotes (market closed or quotes stale). Try during market hours.")
        return
    day = snap["snapshot_date"].iloc[0]
    snap_path = DATA_DIR / f"{args.ticker}_options_{day}.csv"
    snap.to_csv(snap_path, index=False)
    total = append_history(snap, DATA_DIR / f"{args.ticker}_options_history.csv")
    print(f"Saved {len(snap)} call quotes to {snap_path}")
    print(f"History file now holds {total} rows across all snapshots")


if __name__ == "__main__":
    main()