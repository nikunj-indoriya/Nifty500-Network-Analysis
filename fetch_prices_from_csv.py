# fetch_prices_from_csv.py
"""
Fetch daily adjusted close prices for tickers listed in a local Nifty CSV file (Nifty200, Nifty500, etc.)
and save a combined prices DataFrame.

Expected input CSV columns: ['Company Name','Industry','Symbol','Series','ISIN Code']

Outputs:
  data/prices_raw.csv  : combined Adjusted Close prices
  data/failed_tickers.txt : list of tickers that could not be fetched
"""

import pandas as pd
import yfinance as yf
from pathlib import Path
import time, random, datetime as dt

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
csv_path = "ind_nifty500list.csv"    # change here if you want Nifty200 again
years = 3                            # number of years of data
batch_size = 8                       # tickers per batch (8–10 safer for 500 list)
pause = (5, 10)                      # random sleep between batches (seconds)
max_retries = 3                      # retry attempts per batch
outdir = Path("data"); outdir.mkdir(exist_ok=True)
prices_path = outdir / "prices_raw.csv"
failed_file = outdir / "failed_tickers.txt"

# ---------------------------------------------------------------------
# LOAD SYMBOLS
# ---------------------------------------------------------------------
df = pd.read_csv(csv_path)
tickers = df["Symbol"].astype(str).str.strip().tolist()
yahoo_tickers = [t.upper() + ".NS" for t in tickers]
print(f"Loaded {len(yahoo_tickers)} tickers from {csv_path}")

# Date range
start = (dt.date.today() - dt.timedelta(days=int(365 * years))).isoformat()
end = (dt.date.today() + dt.timedelta(days=1)).isoformat()
print(f"Fetching data from {start} to {end}")

# ---------------------------------------------------------------------
# MAIN DOWNLOAD LOOP
# ---------------------------------------------------------------------
prices = pd.DataFrame()
failed = []

for i in range(0, len(yahoo_tickers), batch_size):
    batch = yahoo_tickers[i:i + batch_size]
    print(f"\nBatch {i // batch_size + 1}/{len(yahoo_tickers) // batch_size + 1}: "
          f"{batch[0]} … {batch[-1]} ({len(batch)})")

    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(batch, start=start, end=end,
                               interval="1d", group_by="ticker",
                               progress=False, threads=True, auto_adjust=True)

            if data.empty:
                raise RuntimeError("Empty response")

            # Multi-ticker DataFrame handling
            if isinstance(data.columns, pd.MultiIndex):
                valid = {}
                for t in batch:
                    sub = data.get(t)
                    if sub is not None and not sub.empty:
                        col = 'Adj Close' if 'Adj Close' in sub.columns else 'Close'
                        valid[t] = sub[col]
                if not valid:
                    raise RuntimeError("No valid tickers in batch")
                batch_df = pd.DataFrame(valid)
            else:
                # Single ticker case
                col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                batch_df = data[[col]].rename(columns={col: batch[0]})

            # Merge into cumulative dataframe
            prices = pd.concat([prices, batch_df], axis=1)
            prices = prices[~prices.index.duplicated(keep='first')]
            prices.sort_index(inplace=True)
            prices.to_csv(prices_path)

            print(f"  ✓ Saved {len(batch_df.columns)} tickers (total {prices.shape[1]})")
            break

        except Exception as e:
            print(f"  attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                failed.extend(batch)
            time.sleep(6 + 3 * attempt)
    # pause between batches
    time.sleep(random.uniform(*pause))

# ---------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------
if failed:
    with open(failed_file, "w") as f:
        f.write("\n".join(failed))
    print(f"\nFailed tickers written to {failed_file} ({len(failed)} failures)")

print(f"\nDone. Saved consolidated prices to {prices_path}")
print("Final shape:", prices.shape)
