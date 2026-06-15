"""Week 1 revision: daily sector ETF data.

PyCharm shortcut note:
Settings -> Keymap -> Search for -> Execute Selection in Python Console
Change it to the shortcut you want for running a line or selected code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Read the CSV from the web
# -----------------------------------------------------------------------------

DATA_URL = "https://openbondassetpricing.com/wp-content/uploads/2026/06/sector_etf_panel.csv"

# Output folder.
# This is the folder where the cleaned files will be saved.
OUTPUT_DIR = Path("fins2026") / "week2" / "scratch" / "week1_recap"
# This is a folder inside OUTPUT_DIR where the plots will be saved.
FIGURE_DIR = OUTPUT_DIR / "figures"
# Make OUTPUT_DIR if needed. parents=True also makes missing parent folders.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Make FIGURE_DIR if needed. exist_ok=True avoids an error if it already exists.
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sector_etf_panel = pd.read_csv(DATA_URL)

print("Raw data loaded")
print(sector_etf_panel.head())
print(sector_etf_panel.info())
