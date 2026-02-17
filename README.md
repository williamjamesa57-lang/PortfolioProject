# quant-portfolio

Two side projects I've been building to get more hands-on with quant methods. Both are research/learning exercises — not production trading systems.

---

## Projects

### 1. Statistical Arbitrage Engine
Mean-reversion pairs trading strategy on NYSE's top 50 tickers by volume. Uses the Engle-Granger cointegration test to identify pairs and generate trade signals.

**Status:** data loader + cointegration test done, signal generation WIP

### 2. Earnings Surprise Predictor
Logistic regression model to predict whether a company will beat or miss earnings estimates. Pulls features from SEC filings (10-Q/10-K) using basic NLP.

**Status:** data loader done, NLP extractor in progress

---

## Setup

```bash
pip install -e ".[dev]"
touch .env
mkdir data data/temp
```

You'll need to add SEC EDGAR credentials to `.env` — required by their API policy:

```
SEC_EDGAR_USER_NAME="Firstname Lastname"
SEC_EDGAR_USER_EMAIL="youremail@example.com"
```

> Use a real name and email. SEC will block access for generic/fake credentials.

Download filings:
```bash
python -c "from utils.data_loader import DataLoader; DataLoader().load_sec_filings()"
```

---

## Tests

```bash
# fast, uses cached data (~5s)
pytest tests/ -m "not slow" -v

# hits SEC API directly, run sparingly
pytest tests/ -m slow -v
```

---

## Timeline

| Week | Project 1 (Stat Arb) | Project 2 (Earnings) |
|------|----------------------|----------------------|
| 1 | Data loader pipeline | Data loader pipeline |
| 2 | Cointegration component | NLP extractor component |

---

## Data & License

**Code:** MIT — do whatever you want with it.

**Data:** Not included. You have to download it yourself:
- Prices: `yfinance` (Yahoo Finance ToS — non-commercial)
- Filings: `sec-edgar-downloader` (public domain, but SEC discourages bulk redistribution)

Never commit raw data — it's in `.gitignore` already.

---

## Notes
> 🚀🍸 **NOT FINANCIAL ADVICE** 🍸🚀
- This is a research/learning project. The models output numbers, not money.
- Past results mean nothing. The market will humble you.
- If you YOLO your savings into a pairs trade because a cointegration test said so, that's entirely on you.
- Consult a real financial advisor before touching live capital. A licensed one. Not this repo.