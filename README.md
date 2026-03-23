# sautiNet
Decentralized Sentiment Analysis for Kenyan Social Media Data

How to Run it



Terminal 1 — Backend:
cd sautiNet/sautinet-ml-backend

python -m venv venv && source venv/bin/activate

pip install fastapi uvicorn[standard] pydantic pydantic-settings transformers torch tokenizers sentencepiece httpx aiohttp aiofiles websockets 

python-dotenv python-multipart

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Wait for "SentiKenya is LIVE" — the first run downloads the HuggingFace model (~1-2 GB).



Terminal 2 — Frontend:

cd sautiNet/sautinet-frontend

python -m venv venv && source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
