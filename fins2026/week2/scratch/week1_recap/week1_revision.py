"""Week 1 revision: daily stock data.

PyCharm shortcut note:
Settings -> Keymap -> Search for ->Execute Selection in Python Console
Change it to the shortcut you want for running a line or selected code.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Read the CSV from the web
# -----------------------------------------------------------------------------

DATA_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/06/stock_panel.csv"

# Output folder.
# This is the folder where the cleaned files will be saved.
OUTPUT_DIR = Path("fins2026") / "week2" / "scratch" / "week1_recap"
# This is a folder inside OUTPUT_DIR where the plots will be saved.
FIGURE_DIR = OUTPUT_DIR / "figures"
# Make OUTPUT_DIR if needed. parents=True also makes missing parent folders.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Make FIGURE_DIR if needed. exist_ok=True avoids an error if it already exists.
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Try local parquet first, fall back to local CSV, then web.
CACHED_PARQUET = OUTPUT_DIR / "stock_panel_with_features.parquet"
CACHED_CSV = OUTPUT_DIR / "stock_panel_with_features.csv"

if CACHED_PARQUET.exists():
    try:
        stock_panel = pd.read_parquet(CACHED_PARQUET)
        print(f"Loaded {len(stock_panel):,} rows from local parquet")
    except Exception:
        stock_panel = pd.read_csv(CACHED_CSV)
        print("Parquet read failed; loaded from local CSV")
elif CACHED_CSV.exists():
    stock_panel = pd.read_csv(CACHED_CSV)
    print("Loaded from local CSV")
else:
    stock_panel = pd.read_csv(DATA_URL)
    print("Downloaded raw data from web")

print("Raw data loaded")
print(stock_panel.head())
print(stock_panel.info())


# -----------------------------------------------------------------------------
# 2. Stage 1 of the Data Factory Floor: clean and validate the data
# -----------------------------------------------------------------------------

# Convert the date column from text into a proper pandas datetime column.
stock_panel["date"] = pd.to_datetime(stock_panel["date"])

# Convert every column except date and ticker into numeric values.
# errors="coerce" turns impossible values into missing values, which are easy to find.
numeric_columns = [col for col in stock_panel.columns if col not in ["date", "ticker"]]
stock_panel[numeric_columns] = stock_panel[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce",
)

# Sort the panel so each stock is in calendar order.
stock_panel = stock_panel.sort_values(["ticker", "date"]).reset_index(drop=True)

# Check for duplicate date-ticker observations.
duplicate_rows = stock_panel.duplicated(subset=["date", "ticker"])
number_of_duplicates = duplicate_rows.sum()

print("\nStage 1 checks")
print(f"Rows: {len(stock_panel):,}")
print(f"Columns: {len(stock_panel.columns):,}")
print(f"Duplicate date-ticker rows: {number_of_duplicates:,}")
print("Missing values by column:")
print(stock_panel.isna().sum())

# If duplicates exist, print them
if number_of_duplicates > 0:
    print("\nDuplicate rows:")
    print(stock_panel.loc[duplicate_rows].head(20))

print("\nCleaned data preview")
print(stock_panel.head())


# -----------------------------------------------------------------------------
# 3. Stage 2 of the Data Factory Floor: feature engineering
# -----------------------------------------------------------------------------

# Daily return is the percentage change in close price within each ticker.
stock_panel["daily_return"] = stock_panel.groupby("ticker")["close"].pct_change()

# Dollar volume is the approximate dollar value traded each day.
stock_panel["dollar_volume"] = stock_panel["share_volume"] * stock_panel["close"]

print("\nFeature-engineered panel preview")
print(stock_panel.head(10))


# -----------------------------------------------------------------------------
# 4. Basic performance summary with groupby
# -----------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

# Start with simple daily return statistics for each stock.
return_stats = stock_panel.groupby("ticker")["daily_return"].agg(
    average_daily_return="mean",
    daily_volatility="std",
    observations="count",
)

# Convert daily statistics into common annualized statistics.
return_stats["average_daily_return_pct"] = (
    return_stats["average_daily_return"] * 100
)
return_stats["annualized_return_pct"] = (
    return_stats["average_daily_return"] * TRADING_DAYS_PER_YEAR * 100
)
return_stats["daily_volatility_pct"] = return_stats["daily_volatility"] * 100
return_stats["annualized_volatility_pct"] = (
    return_stats["daily_volatility"] * (TRADING_DAYS_PER_YEAR**0.5) * 100
)

# Risk-free rate is assumed to be zero, so the Sharpe ratio is return divided by risk.
return_stats["sharpe_ratio_rf_0"] = (
    return_stats["average_daily_return"]
    / return_stats["daily_volatility"]
    * (TRADING_DAYS_PER_YEAR**0.5)
)

# Keep only the summary columns.
return_summary = return_stats[
    [
        "observations",
        "average_daily_return_pct",
        "annualized_return_pct",
        "daily_volatility_pct",
        "annualized_volatility_pct",
        "sharpe_ratio_rf_0",
    ]
].round(3)

print("\nReturn summary")
print(return_summary)


# -----------------------------------------------------------------------------
# 5. Data wrangling: pivot_table for wide close prices
# -----------------------------------------------------------------------------

# Long format has one row per date-ticker pair.
print("\nLong format shape")
print(stock_panel.shape)
print(stock_panel.head())

# Wide format puts each ticker into its own column.
close_wide = stock_panel.pivot_table(
    index="date",
    columns="ticker",
    values="close",
)

print("\nWide close-price table")
print(close_wide.head())
print(close_wide.shape)

# In wide format, pct_change computes returns down each ticker column.
returns_wide = close_wide.pct_change()

print("\nWide return table")
print(returns_wide.head())


# -----------------------------------------------------------------------------
# 6. Why long format is usually better for storing data
# -----------------------------------------------------------------------------

# Wide format is convenient for matrix calculations.
# Long format is better for storage because each row has the same meaning:
# one date, one ticker, and one set of variables.

# Example: the long panel can store close, volume, return, and dollar volume together.
print("\nLong format keeps multiple variables neatly together")
example_columns = [
    "date",
    "ticker",
    "close",
    "share_volume",
    "daily_return",
    "dollar_volume",
]
print(stock_panel[example_columns].head())

# If we only need wide returns for a calculation, we can create them temporarily.
# We do not need to permanently store the whole dataset in wide format.

# Convert the wide returns back to long format when we want a tidy panel again.
returns_long = returns_wide.reset_index().melt(
    id_vars="date",
    var_name="ticker",
    value_name="daily_return_from_wide",
)

print("\nReturns converted back to long format")
print(returns_long.head())


# -----------------------------------------------------------------------------
# 7. Save the feature-engineered panel for later use
# -----------------------------------------------------------------------------

OUTPUT_CSV = OUTPUT_DIR / "stock_panel_with_features.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "stock_panel_with_features.parquet"

stock_panel.to_csv(OUTPUT_CSV, index=False)
stock_panel.to_parquet(OUTPUT_PARQUET, index=False)

print("\nSaved feature-engineered files")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)


# -----------------------------------------------------------------------------
# 8. Simple time-series and bar plots
# -----------------------------------------------------------------------------

# Plot 1: daily returns through time.
daily_returns_pct = returns_wide * 100

plt.figure(figsize=(10, 6))  # Start a new 10-by-6 inch figure.
daily_returns_pct.plot(ax=plt.gca())  # plt.gca() gets the current axes, or chart area.
plt.title("Daily returns")  # Add the chart title.
plt.xlabel("Date")  # Label the horizontal axis.
plt.ylabel("Daily return (%)")  # Label the vertical axis.
plt.legend(title="Ticker")  # Add a legend to identify each ticker.
plt.tight_layout()  # Improve spacing around titles and labels.
plt.savefig(FIGURE_DIR / "01_daily_returns.png", dpi=150)  # Save the chart.
plt.show()  # Display the chart on screen.
plt.close()  # Close the chart before making the next one.


# Plot 2: cumulative returns through time.
cumulative_returns = (1 + returns_wide.fillna(0)).cumprod() - 1
cumulative_returns_pct = cumulative_returns * 100

plt.figure(figsize=(10, 6))  # Start a new 10-by-6 inch figure.
cumulative_returns_pct.plot(ax=plt.gca())  # plt.gca() gets the current axes, or chart area.
plt.title("Cumulative returns")  # Add the chart title.
plt.xlabel("Date")  # Label the horizontal axis.
plt.ylabel("Cumulative return (%)")  # Label the vertical axis.
plt.legend(title="Ticker")  # Add a legend to identify each ticker.
plt.tight_layout()  # Improve spacing around titles and labels.
plt.savefig(FIGURE_DIR / "02_cumulative_returns.png", dpi=150)  # Save the chart.
plt.show()  # Display the chart on screen.
plt.close()  # Close the chart before making the next one.


# Plot 3: annualized mean return.
plt.figure(figsize=(8, 5))  # Start a new 8-by-5 inch figure.
return_summary["annualized_return_pct"].plot(kind="bar", ax=plt.gca())  # Bars on current axes.
plt.title("Annualized mean return")  # Add the chart title.
plt.xlabel("Ticker")  # Label the horizontal axis.
plt.ylabel("Annualized return (%)")  # Label the vertical axis.
plt.xticks(rotation=0)  # Keep ticker labels horizontal.
plt.tight_layout()  # Improve spacing around titles and labels.
plt.savefig(FIGURE_DIR / "03_annualized_mean_return.png", dpi=150)  # Save the chart.
plt.show()  # Display the chart on screen.
plt.close()  # Close the chart before making the next one.


# Plot 4: annualized volatility.
plt.figure(figsize=(8, 5))  # Start a new 8-by-5 inch figure.
return_summary["annualized_volatility_pct"].plot(kind="bar", ax=plt.gca())  # Bars on current axes.
plt.title("Annualized volatility")  # Add the chart title.
plt.xlabel("Ticker")  # Label the horizontal axis.
plt.ylabel("Annualized volatility (%)")  # Label the vertical axis.
plt.xticks(rotation=0)  # Keep ticker labels horizontal.
plt.tight_layout()  # Improve spacing around titles and labels.
plt.savefig(FIGURE_DIR / "04_annualized_volatility.png", dpi=150)  # Save the chart.
plt.show()  # Display the chart on screen.
plt.close()  # Close the chart before making the next one.


# Plot 5: Sharpe ratio with risk-free rate equal to zero.
plt.figure(figsize=(8, 5))  # Start a new 8-by-5 inch figure.
return_summary["sharpe_ratio_rf_0"].plot(kind="bar", ax=plt.gca())  # Bars on current axes.
plt.title("Sharpe ratio")  # Add the chart title.
plt.xlabel("Ticker")  # Label the horizontal axis.
plt.ylabel("Sharpe ratio")  # Label the vertical axis.
plt.xticks(rotation=0)  # Keep ticker labels horizontal.
plt.tight_layout()  # Improve spacing around titles and labels.
plt.savefig(FIGURE_DIR / "05_sharpe_ratio.png", dpi=150)  # Save the chart.
plt.show()  # Display the chart on screen.
plt.close()  # Close the chart.
print("\nSaved plots")
print(FIGURE_DIR / "01_daily_returns.png")
print(FIGURE_DIR / "02_cumulative_returns.png")
print(FIGURE_DIR / "03_annualized_mean_return.png")
print(FIGURE_DIR / "04_annualized_volatility.png")
print(FIGURE_DIR / "05_sharpe_ratio.png")



