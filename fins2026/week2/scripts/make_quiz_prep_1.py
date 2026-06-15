"""Generate quiz_prep_1.pdf — 20 MCQ quiz on the crypto panel.

Uses the same data and workflow as crypto_narrative.py.
Output goes to fins2026/week2/tests/quiz_prep_1.pdf.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fpdf import FPDF

DATA_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/06/crypto_panel.csv"
CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "data"
CACHE_PATH = CACHE_DIR / "crypto_panel.csv"

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "quiz_prep_1.pdf"

# ── data (cached) ─────────────────────────────────────────────────────────
if CACHE_PATH.exists():
    panel = pd.read_csv(CACHE_PATH)
else:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, CACHE_PATH)
    panel = pd.read_csv(CACHE_PATH)
    print(f"Cached: {CACHE_PATH}")

panel = panel.copy()
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
# returns
panel["ret"] = panel.groupby("ticker")["close"].pct_change()
# pivot
pivot = panel.pivot(index="date", columns="ticker", values="close")
rets = pivot.pct_change()
growth = (1 + rets).cumprod()

n_tickers = panel["ticker"].nunique()
n_dates = panel["date"].nunique()
n_rows, n_cols = panel.shape

# helper: round to 4 d.p. for reproducibility in questions
r4 = lambda v: round(v, 4)

# ── questions ─────────────────────────────────────────────────────────────

questions: list[dict] = []

# Q1 — shape
questions.append(
    {
        "q": "What is the shape (rows, columns) of the raw crypto panel after loading with pd.read_csv?",
        "opts": [
            "(2351, 5)",
            "(11755, 7)",
            "(11755, 8)",
            "(2351, 7)",
        ],
        "ans": "B",
    }
)

# Q2 — columns
questions.append(
    {
        "q": "Which of the following is NOT a column in the crypto panel?",
        "opts": [
            "close",
            "usd_volume",
            "adjusted_close",
            "high",
        ],
        "ans": "C",
    }
)

# Q3 — ticker count
questions.append(
    {
        "q": "How many unique tickers are in the crypto panel?",
        "opts": ["3", "4", "5", "7"],
        "ans": "C",
    }
)

# Q4 — date range
questions.append(
    {
        "q": "What is the date range of the crypto panel?",
        "opts": [
            "2020-01-01 to 2026-06-09",
            "2019-01-01 to 2025-12-31",
            "2020-01-01 to 2025-12-31",
            "2021-01-01 to 2026-06-09",
        ],
        "ans": "A",
    }
)

# Q5 — missing values
questions.append(
    {
        "q": "How many missing (null) values are in the raw crypto panel?",
        "opts": ["0", "5", "11755", "2351"],
        "ans": "A",
    }
)

# Q6 — rows per ticker
questions.append(
    {
        "q": "How many observations (rows) does each ticker have in the panel?",
        "opts": ["2351", "11755", "4702", "5"],
        "ans": "A",
    }
)

# Q7 — pivot shape
row_count, col_count = pivot.shape
q7_right = f"({row_count}, {col_count})"
questions.append(
    {
        "q": 'After running panel.pivot(index="date", columns="ticker", values="close"), what is the shape of the resulting DataFrame?',
        "opts": [
            q7_right,
            "(5, 2351)",
            "(11755, 5)",
            "(2351, 7)",
        ],
        "ans": "A",
    }
)

# Q8 — returns NaN count
na_ret_count = int(panel["ret"].isna().sum())
questions.append(
    {
        "q": f"When you run .pct_change() on the close column grouped by ticker, how many NaN values appear in the new returns column for the full panel?",
        "opts": [str(na_ret_count), "0", "11755", "2351"],
        "ans": "A",
    }
)

# Q9 — volatility (std of BTC returns)
btc_std = r4(panel.loc[panel["ticker"] == "BTC", "ret"].std())
distractor_a = r4(panel.loc[panel["ticker"] == "ADA", "ret"].std())
distractor_e = r4(panel.loc[panel["ticker"] == "ETH", "ret"].std())
distractor_d = r4(panel.loc[panel["ticker"] == "DOGE", "ret"].std())
questions.append(
    {
        "q": "What is the standard deviation of BTC daily returns (approximately)?",
        "opts": [
            str(distractor_a),
            str(btc_std),
            str(distractor_e),
            str(distractor_d),
        ],
        "ans": "B",
    }
)

# Q10 — DOGE mean return
doge_mean = r4(panel.loc[panel["ticker"] == "DOGE", "ret"].mean())
questions.append(
    {
        "q": "Which ticker has the highest mean daily return?",
        "opts": [
            "ADA",
            "BTC",
            "DOGE",
            "ETH",
        ],
        "ans": "C",
    }
)

# Q11 — cumulative growth of $1
final_growth = growth.iloc[-1]
doge_growth = r4(final_growth["DOGE"])
questions.append(
    {
        "q": "What is the final cumulative growth-of-$1 value for DOGE?",
        "opts": [
            str(doge_growth),
            str(r4(final_growth["ADA"])),
            str(r4(final_growth["ETH"])),
            str(r4(final_growth["LINK"])),
        ],
        "ans": "A",
    }
)

# Q12 — highest correlation
corr = pivot.corr()
max_pair = (
    corr.where(~corr.isin([1.0]))
    .stack()
    .idxmax()
)
max_val = r4(corr.loc[max_pair[0], max_pair[1]])
questions.append(
    {
        "q": f"Which pair of crypto assets has the highest correlation in the panel (to 4 d.p.)?",
        "opts": [
            f"ADA-LINK ({max_val})",
            f"ETH-BTC ({r4(corr.loc['ETH','BTC'])})",
            f"DOGE-ETH ({r4(corr.loc['DOGE','ETH'])})",
            f"ADA-DOGE ({r4(corr.loc['ADA','DOGE'])})",
        ],
        "ans": "A",
    }
)

# Q13 — ADA-BTC correlation
ada_btc = r4(corr.loc["ADA", "BTC"])
questions.append(
    {
        "q": "What is the correlation between ADA and BTC daily returns (to 4 d.p.)?",
        "opts": [
            str(ada_btc),
            str(r4(corr.loc["ADA", "LINK"])),
            str(r4(corr.loc["BTC", "ETH"])),
            str(r4(corr.loc["BTC", "DOGE"])),
        ],
        "ans": "A",
    }
)

# Q14 — Sharpe ratio
ann_sharpe = rets.mean() / rets.std() * (252 ** 0.5)
top_sharpe_ticker = ann_sharpe.idxmax()
top_sharpe_val = r4(ann_sharpe.max())
questions.append(
    {
        "q": f"Assuming a 0% risk-free rate and 252 trading days, which ticker has the highest annualized Sharpe ratio?",
        "opts": [
            f"ETH ({r4(ann_sharpe['ETH'])})",
            f"BTC ({r4(ann_sharpe['BTC'])})",
            f"DOGE ({r4(ann_sharpe['DOGE'])})",
            f"ADA ({r4(ann_sharpe['ADA'])})",
        ],
        "ans": "A",
    }
)

# Q15 — mean volume for BTC
btc_mean_vol = int(panel.loc[panel["ticker"] == "BTC", "usd_volume"].mean())
ada_mean_vol = int(panel.loc[panel["ticker"] == "ADA", "usd_volume"].mean())
questions.append(
    {
        "q": "What is the mean usd_volume for BTC (rounded to the nearest billion)?",
        "opts": [
            "~18.8 billion",
            "~36.7 billion",
            "~1.3 billion",
            "~125.9 billion",
        ],
        "ans": "B",
    }
)

# Q16 — code output: string interpolation
questions.append(
    {
        "q": "What does the expression f'Shape: {panel.shape}' print?",
        "opts": [
            "Shape: 11755, 7",
            "Shape: (11755, 7)",
            "Shape: (2351, 5)",
            "Shape: (2351, 7)",
        ],
        "ans": "B",
    }
)

# Q17 — dropna shape
clean_shape = rets.dropna().shape
questions.append(
    {
        "q": f'After running returns = pivot.pct_change() and then returns.dropna(), what is the shape of the result?',
        "opts": [
            f"({clean_shape[0]}, {clean_shape[1]})",
            f"({clean_shape[0] + 1}, {clean_shape[1]})",
            f"({clean_shape[0] - 1}, {clean_shape[1]})",
            f"({clean_shape[0]}, {clean_shape[1] - 1})",
        ],
        "ans": "A",
    }
)

# Q18 — BTC max close
btc_max = r4(panel.loc[panel["ticker"] == "BTC", "close"].max())
questions.append(
    {
        "q": "What is the maximum close price for BTC in the panel?",
        "opts": [
            str(btc_max),
            str(r4(panel.loc[panel["ticker"] == "BTC", "close"].median())),
            str(r4(panel.loc[panel["ticker"] == "BTC", "close"].mean())),
            str(r4(panel.loc[panel["ticker"] == "ETH", "close"].max())),
        ],
        "ans": "A",
    }
)

# Q19 — mean close for ETH
eth_mean = r4(panel.loc[panel["ticker"] == "ETH", "close"].mean())
questions.append(
    {
        "q": "What is the mean close price for ETH (approximately)?",
        "opts": [
            str(eth_mean),
            str(r4(panel.loc[panel["ticker"] == "ADA", "close"].mean())),
            str(r4(panel.loc[panel["ticker"] == "LINK", "close"].mean())),
            str(r4(panel.loc[panel["ticker"] == "ETH", "close"].median())),
        ],
        "ans": "A",
    }
)

# Q20 — which ticker won the growth race
winner = final_growth.idxmax()
winner_val = r4(final_growth.max())
questions.append(
    {
        "q": "Which ticker has the highest cumulative growth of $1 over the full sample?",
        "opts": [
            f"DOGE (${winner_val})",
            f"ETH (${r4(final_growth['ETH'])})",
            f"BTC (${r4(final_growth['BTC'])})",
            f"ADA (${r4(final_growth['ADA'])})",
        ],
        "ans": "A",
    }
)

assert len(questions) == 20, f"Expected 20 questions, got {len(questions)}"


# ── PDF generation ────────────────────────────────────────────────────────

class QuizPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "FINS 2026 - Quiz 1 Practice: Crypto Panel", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


pdf = QuizPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# ── Instructions ──────────────────────────────────────────────────────────
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 6, "Instructions", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 4.5,
    "This quiz uses the daily crypto panel loaded from:\n"
    "  https://openbondassetpricing.com/wp-content/uploads/2026/06/crypto_panel.csv\n"
    "You may reproduce the analysis steps in Python code to answer the questions. "
    "Each question has exactly one correct answer.",
)
pdf.ln(3)

# ── Data reference box ────────────────────────────────────────────────────
pdf.set_font("Courier", "", 7.5)
ref = (
    "Panel shape: (11755, 7)  |  Columns: date, ticker, open, high, low, close, usd_volume\n"
    "Tickers: ADA, BTC, DOGE, ETH, LINK  |  Observations per ticker: 2351\n"
    "Date range: 2020-01-01 to 2026-06-09  |  Missing values: 0\n"
    "After pivot('date', 'ticker', 'close'): (2351, 5)  |  Returns via .pct_change()\n"
    "Cumulative growth via (1 + rets).cumprod()  |  Sharpe: 252 trading days, rf = 0%"
)
pdf.set_fill_color(240, 240, 240)
pdf.multi_cell(0, 3.8, ref, fill=True)
pdf.ln(4)

# ── Questions ─────────────────────────────────────────────────────────────
for i, qdata in enumerate(questions, 1):
    # Check if we need a new page (rough heuristic)
    if pdf.get_y() > 250:
        pdf.add_page()

    pdf.set_font("Helvetica", "B", 9.5)
    pdf.multi_cell(0, 5, f"{i:2d}. {qdata['q']}")

    pdf.set_font("Helvetica", "", 9)
    labels = ["A", "B", "C", "D"]
    for label, opt in zip(labels, qdata["opts"]):
        pdf.cell(10, 5, f"  {label}.")
        pdf.cell(0, 5, f" {opt}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

# ── Answer key ────────────────────────────────────────────────────────────
pdf.add_page()
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Answer Key", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)

answers = [q["ans"] for q in questions]
pdf.set_font("Courier", "", 9)
for i, ans in enumerate(answers, 1):
    line = f"  {i:2d}.  {ans}"
    if i % 2 == 0:
        pdf.cell(0, 5.5, line, new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 5.5, line)
if len(questions) % 2 == 1:
    pdf.ln(5.5)

pdf.ln(4)
pdf.set_font("Helvetica", "I", 8)
pdf.multi_cell(0, 4,
    "Tip: Re-run the analysis step by step in a Python script or Jupyter notebook to "
    "verify each answer. The goal is to confirm you can produce every number yourself."
)

pdf.output(str(OUTPUT_PATH))
print(f"Wrote: {OUTPUT_PATH}")
print(f"  Questions: {len(questions)}")
print(f"  Pages: {pdf.page_no()}")
