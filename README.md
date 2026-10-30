# MATH-1360-Proj: Black-Scholes on ASML and SMIC

Authors: Gabriel Peralta, Lucas Respicio

## Goal
Apply the Black-Scholes model (GBM SDE) to European options on ASML
(Euronext Amsterdam) and extend the analysis to SMIC.

## Layout
- src/data.py: download prices (yfinance), compute log returns
- src/fit.py: estimate mu and sigma, normality checks
- src/simulate.py: GBM paths (exact and Euler-Maruyama)
- src/blackscholes.py: price, Greeks, implied volatility
- data/: price CSVs and hand-collected option quotes
- figures/: saved plots
- report/: drafts
