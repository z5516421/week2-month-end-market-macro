# Crypto Report Agent Context

Scope: this context applies only inside `fins2026/week2/scratch/crypto_report`.

## Reporting Rules

- Keep the analysis self-contained in this folder; do not modify context files outside it.
- Treat `data/raw_crypto_panel.csv` as the cached source extract.
- Use `code/crypto_narrative.py` as the reproducible build script copy.
- Report returns, volatility, drawdowns, and correlations in percent terms where applicable.
- Round final table values to 3 decimal places.
- Keep table column names short enough for Word; avoid labels longer than two lines.
- Use a zero risk-free rate unless the senior partner explicitly changes the mandate.
- Keep figure notes short; do not let text below figures overrun the image.
- Prefer FT-style charts with direct, decision-useful titles.

## Rebuild Command

```bash
./.venv/bin/python fins2026/week2/scratch/crypto_narrative.py
```
