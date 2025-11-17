"""
step2_clean_returns_corr.py
------------------------------------------------------
Cleans price data (from Nifty500 fetch), computes log returns,
and saves cleaned prices, returns, and correlation matrix
for subsequent network analysis.

Input :
  data/prices_raw.csv      (from fetch_prices_from_csv.py)
Output:
  data/prices_clean.csv
  data/returns.csv
  data/corr_matrix.csv
  data/plots/corr_distribution.png
  data/plots/corr_heatmap.png
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ CONFIG ------------------
data_dir = Path("data")
plots_dir = data_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

input_file = data_dir / "prices_raw.csv"
output_prices = data_dir / "prices_clean.csv"
output_returns = data_dir / "returns.csv"
output_corr = data_dir / "corr_matrix.csv"

# Parameters
nan_threshold = 0.20   # drop tickers with >20% missing data
fill_method = True     # forward/backward fill small gaps

# ------------------ LOAD --------------------
print(f"Loading {input_file} ...")
df = pd.read_csv(input_file, index_col=0, parse_dates=True)
print("Raw shape:", df.shape)

# Remove duplicated dates and sort
df = df[~df.index.duplicated(keep="first")]
df = df.sort_index()

# ------------------ CLEANING -----------------
# 1. Drop invalid or placeholder tickers
invalid_cols = [c for c in df.columns if "DUMMY" in c.upper() or c.strip() == ""]
if invalid_cols:
    print("Dropping invalid columns:", invalid_cols)
    df = df.drop(columns=invalid_cols)

# 2. Drop columns completely empty
before_cols = df.shape[1]
df = df.dropna(axis=1, how="all")
print(f"Dropped {before_cols - df.shape[1]} completely empty columns.")

# 3. Remove tickers with too many NaNs (>nan_threshold)
nan_ratio = df.isna().mean()
too_many_nans = nan_ratio[nan_ratio > nan_threshold].index.tolist()
if too_many_nans:
    print(f"Dropping {len(too_many_nans)} tickers with >{int(nan_threshold*100)}% NaNs.")
    df = df.drop(columns=too_many_nans)
else:
    print("No tickers exceeded NaN threshold.")

# 4. Fill small gaps
if fill_method:
    df = df.ffill(limit=5).bfill(limit=5)

# 5. Drop rows with all NaNs (no price data for that date)
df = df.dropna(how="all")

# 6. Save cleaned prices
df.to_csv(output_prices)
print(f"Saved cleaned prices to {output_prices} with shape {df.shape}")

# ------------------ RETURNS ------------------
# Compute daily log returns
returns = np.log(df / df.shift(1))
returns = returns.dropna(how="all")

# Drop tickers with all-zero or constant values (flat series)
flat = returns.columns[(returns.var() == 0)]
if len(flat) > 0:
    print(f"Dropping {len(flat)} flat tickers with zero variance.")
    returns = returns.drop(columns=flat)

# Save returns
returns.to_csv(output_returns)
print(f"Saved log returns to {output_returns} with shape {returns.shape}")

# ------------------ CORRELATIONS -------------
corr_matrix = returns.corr(method="pearson")
corr_matrix.to_csv(output_corr)
print(f"Saved correlation matrix to {output_corr} with shape {corr_matrix.shape}")

# ------------------ SUMMARY ------------------
print("\nSummary:")
print(f"Final tickers: {len(df.columns)}")
print(f"Date range: {df.index.min().date()} → {df.index.max().date()}")
print(f"Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix,1)].mean():.3f}")
print("Correlation matrix saved. Ready for network construction.")

# ------------------ OPTIONAL VISUALS ---------
plt.figure(figsize=(7,5))
plt.hist(corr_matrix.values.flatten(), bins=60, color="steelblue", alpha=0.8)
plt.title("Distribution of Pairwise Correlations (Nifty 500)")
plt.xlabel("Correlation coefficient")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(plots_dir / "corr_distribution.png", dpi=150)
plt.close()

# Correlation heatmap (sample 50x50 subset for clarity)
subset = corr_matrix.sample(n=min(50, len(corr_matrix)), axis=0).sample(n=min(50, len(corr_matrix)), axis=1)
plt.figure(figsize=(8,6))
sns.heatmap(subset, cmap="coolwarm", center=0, square=True, cbar_kws={'label': 'Correlation'})
plt.title("Correlation Heatmap (sample subset)")
plt.tight_layout()
plt.savefig(plots_dir / "corr_heatmap.png", dpi=150)
plt.close()

print("Saved correlation distribution and heatmap plots in data/plots/")
