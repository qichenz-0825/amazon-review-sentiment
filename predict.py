"""Predict review sentiment with the saved model.

Usage:
    python predict.py "Absolutely delicious, will buy again!" "Stale and overpriced"
"""
import argparse

import joblib

from preprocess import clean_text


def load(path="sentiment_model.joblib"):
    return joblib.load(path)


def predict(texts, model):
    probs = model["pipeline"].predict_proba([clean_text(t) for t in texts])[:, 1]
    return [("positive" if p >= model["threshold"] else "negative", float(p)) for p in probs]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("texts", nargs="+", help="one or more review summaries")
    parser.add_argument("--model", default="sentiment_model.joblib")
    args = parser.parse_args()

    model = load(args.model)
    for text, (label, p) in zip(args.texts, predict(args.texts, model)):
        print(f"{label:<8}  P(positive)={p:.3f}  {text}")


if __name__ == "__main__":
    main()
