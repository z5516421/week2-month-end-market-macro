# Crypto Report

This folder contains the Week 2 crypto investment performance report.

## Structure

- `data/` stores the cached raw panel, cleaned panel, close-price matrix, and return matrix.
- `code/` stores a copy of the executable analysis script used to build the report.
- `output/figures/` stores Word-ready FT-style PNG/PDF figures and caption sidecars.
- `output/tables/` stores final CSV tables rounded for reporting.
- `report/` stores the Word `.docx` report.

## Rebuild

From the repository root:

```bash
./.venv/bin/python fins2026/week2/scratch/crypto_narrative.py
```

The script assumes a zero risk-free rate and annualises daily crypto returns with
`365` calendar days per year.
