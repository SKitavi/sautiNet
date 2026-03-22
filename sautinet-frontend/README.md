# SautiNet Frontend — Streamlit Dashboard

## Setup

```bash
cd sautinet-frontend
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501

Make sure the backend is running first:

```bash
cd ../sautinet-ml-backend
uvicorn app.main:app --reload --port 8000
```

## Pages

- **Dashboard** — KPI metrics, sentiment donut, language breakdown, trending topics
- **Analyze Text** — Single text + batch analysis with gauge chart and entity table
- **Counties** — Sentiment bar chart and table for all 47 counties
- **Live Feed** — Real-time WebSocket stream of analyzed posts
