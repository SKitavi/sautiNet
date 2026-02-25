#!/usr/bin/env python3
"""
SentiKenya — Run Everything
==============================
Single entry point that:
  1. Trains the custom BiLSTM model (if not already trained)
  2. Fine-tunes the transformer model (if not already fine-tuned)
  3. Starts the API server with Reddit ingestion

Usage:
  python run.py              # Train models (if needed) + start server
  python run.py --train      # Force retrain both models + start server
  python run.py --server     # Skip training, just start server
  python run.py --train-only # Train models only, don't start server

Requirements:
  pip install fastapi uvicorn transformers torch scikit-learn
  pip install datasets accelerate sentencepiece httpx
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).parent
os.chdir(ROOT)


def banner(text: str):
    print(f"\n{'═' * 65}")
    print(f"  {text}")
    print(f"{'═' * 65}\n")


def model_exists(path: str) -> bool:
    """Check if a trained model exists at the given path."""
    p = ROOT / path
    return p.is_dir() and (p / "config.json").exists()


def train_custom_bilstm(force: bool = False):
    """
    Train the custom BiLSTM + Self-Attention model.
    ~290K params, trains in ~30 seconds on CPU.
    Saves to: ./models/custom-bilstm-v1/
    """
    model_path = "models/custom-bilstm-v1"

    if model_exists(model_path) and not force:
        print(f"  ✓ Custom BiLSTM already trained at ./{model_path}/")
        print(f"    (use --train to force retrain)")
        return True

    banner("TRAINING CUSTOM BiLSTM + ATTENTION MODEL")
    print("  Architecture: Embedding → BiLSTM(2 layers) → Self-Attention → FC")
    print("  Parameters:   ~290K")
    print("  Dataset:      150 labeled Kenyan social media posts")
    print("  Languages:    English, Swahili, Sheng")
    print()

    start = time.time()
    result = subprocess.run(
        [sys.executable, "app/ml/custom_model.py"],
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✓ Custom BiLSTM trained in {elapsed:.1f}s")
        print(f"    Saved to: ./{model_path}/")
        return True
    else:
        print(f"\n  ✗ Custom BiLSTM training failed (exit code {result.returncode})")
        return False


def train_finetuned_transformer(force: bool = False):
    """
    Fine-tune the XLM-RoBERTa transformer on Kenyan data.
    ~278M params (only top layers trained), ~2 min on CPU.
    Downloads base model (~1.1GB) on first run.
    Saves to: ./models/sentikenya-v1/best/
    """
    model_path = "models/sentikenya-v1/best"

    if model_exists(model_path) and not force:
        print(f"  ✓ Fine-tuned transformer already exists at ./{model_path}/")
        print(f"    (use --train to force retrain)")
        return True

    banner("FINE-TUNING XLM-RoBERTa TRANSFORMER")
    print("  Base model:   cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual")
    print("  Strategy:     Freeze layers 0-8, train layers 9-11 + classifier head")
    print("  Parameters:   21.8M trainable (7.9% of 278M total)")
    print("  Dataset:      Same 150 samples")
    print("  NOTE:         First run downloads ~1.1GB model from HuggingFace")
    print()

    start = time.time()
    result = subprocess.run(
        [sys.executable, "app/ml/training/finetune.py"],
        cwd=str(ROOT),
    )
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✓ Transformer fine-tuned in {elapsed:.1f}s")
        print(f"    Saved to: ./{model_path}/")
        return True
    else:
        print(f"\n  ✗ Transformer fine-tuning failed (exit code {result.returncode})")
        return False


def start_server():
    """
    Start the FastAPI server.
    Auto-loads whichever models are available:
      - Fine-tuned transformer (if trained)
      - Custom BiLSTM (if trained)
      - Falls back to rule-based lexicon (always available)
    Reddit ingestion starts automatically.
    """
    banner("STARTING SENTIKENYA SERVER")

    # Show what will load
    models = []
    if model_exists("models/sentikenya-v1/best"):
        models.append("Fine-tuned XLM-RoBERTa (278M params)")
    if model_exists("models/custom-bilstm-v1"):
        models.append("Custom BiLSTM + Attention (290K params)")
    models.append("Rule-based Sheng lexicon (always active)")

    print("  Models that will load:")
    for m in models:
        print(f"    • {m}")

    print()
    print("  Endpoints:")
    print("    POST /api/v1/analyze          — Analyze text sentiment")
    print("    GET  /api/v1/ingestion/status  — Reddit connector status")
    print("    GET  /api/v1/health            — Pipeline health check")
    print("    GET  /api/v1/trending          — Trending topics")
    print("    WS   /ws/feed                  — Real-time sentiment stream")
    print("    GET  /docs                     — Swagger UI")
    print()
    print("  Reddit ingestion: r/Kenya, r/Nairobi (polls every 2 min)")
    print()
    print("  Starting server on http://0.0.0.0:8000 ...")
    print("  Press Ctrl+C to stop")
    print()

    os.execvp(
        sys.executable,
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="SentiKenya — Train models and start the sentiment analysis server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py              # Train (if needed) + start server
  python run.py --train      # Force retrain all models + start
  python run.py --server     # Just start server (skip training)
  python run.py --train-only # Train models only
        """,
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Force retrain all models even if they exist",
    )
    parser.add_argument(
        "--server", action="store_true",
        help="Skip training, just start the server",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Train models but don't start the server",
    )

    args = parser.parse_args()

    banner("SENTIKENYA — Kenyan Sentiment Analysis Platform")
    print("  3-model ensemble: Transformer + Custom BiLSTM + Lexicon")
    print("  Reddit ingestion: r/Kenya, r/Nairobi")
    print("  Languages: English, Swahili, Sheng")

    if not args.server:
        # ── Train models ──
        ok1 = train_custom_bilstm(force=args.train)
        ok2 = train_finetuned_transformer(force=args.train)

        if not ok1 and not ok2:
            print("\n  ⚠ Both models failed to train. Server will use rule-based only.")
        elif not ok1:
            print("\n  ⚠ Custom BiLSTM failed. Server will use transformer + lexicon.")
        elif not ok2:
            print("\n  ⚠ Transformer failed. Server will use BiLSTM + lexicon.")

    if args.train_only:
        banner("TRAINING COMPLETE")
        print("  Run 'python run.py --server' to start the server")
        return

    # ── Start server ──
    start_server()


if __name__ == "__main__":
    main()
