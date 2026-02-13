# Portfolio Project

## Project 1: Statistical Arbitrage Engine

## Project 2: Earning Surprise Predictor

# Timeline
- **Week 1: Data Loader Pipeline**
  - **Project 1:** Statistical Arbitrage Engine (data loader component)
  - **Project 2:** Earning Surprise Predictor (data loader component)

## Setup

```bash
# Install dependencies
pip install -e ".[dev]"

# setup env
touch .env

#create data directory
mkdir data data/temp

#add this to the .env file excluded innitially due to .gitignore and sensitive information safety
#used for SEC EDGAR credentials (REQUIREMENT by SEC policy)
SEC_EDGAR_USER_NAME="firstname lastname"
# do not use generic email. Use valid email or your access to the EDGAR SEC API will be denied
SEC_EDGAR_USER_EMAIL="youremail@.example.com"

+ # Fast unit tests (uses cached data - runs in <5s)
+ pytest tests/ -m "not slow" -v
+ 
+ # Integration tests (hits SEC API - run weekly)
+ pytest tests/ -m slow -v

# Download SEC filings (creates data/sec_filings/)
python -c "from utils.data_loader import DataLoader; DataLoader().load_sec_filings()"
```

## License

- **Code**: [MIT License](LICENSE) — free to use/modify
- **Data**: NOT included in this repo per source terms of data providers:
  - NYSE prices: Downloaded via `yfinance` (Yahoo Finance ToS applies — non-commercial use only)
  - SEC filings: Downloaded via `sec-edgar-downloader` (U.S. government public domain, but bulk redistribution discouraged per SEC guidelines)
- **Never commit raw data** — see `.gitignore`. Users must run download scripts locally.

## NOTES
- Findings that might be extracted from the two models are not financial advice and is not guaranteed to hold for future outcomes
- Trading Financial Assets involves risk of loss
- Consult a Registered Financial Advisor / Expert before trading real capital