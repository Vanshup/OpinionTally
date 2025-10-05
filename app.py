from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re


app = Flask(__name__)

# ✅ Load the pretrained HuggingFace model (Twitter sentiment)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


# Load tokenizer and model once at startup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Class labels based on model training
LABELS = ["negative", "neutral", "positive"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True)
    text_input = ""
    if data:
        text_input = data.get("text", "") or ""

    text_input = text_input.strip()
    if not text_input:
        return jsonify({"error": "Input is empty. Please enter some text."}), 400

    # Split text by lines
    tweets = [line.strip() for line in text_input.splitlines() if line.strip()]

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    results = []

    def clean_tweet(tweet):
        tweet = tweet.lower()
        tweet = re.sub(r"http\S+", "", tweet)   # remove URLs
        tweet = re.sub(r"@\w+", "", tweet)      # remove mentions
        tweet = re.sub(r"#", "", tweet)         # remove hashtags symbol
        tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)  # remove special chars, numbers
        tweet = tweet.strip()
        return tweet


    for tweet in tweets:
        tweet = clean_tweet(tweet)
        # Tokenize tweet
        inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = outputs.logits[0].detach().numpy()
            probs = np.exp(scores) / np.sum(np.exp(scores))
            label_id = int(np.argmax(probs))
            sentiment = LABELS[label_id]
            confidence = float(probs[label_id])

        sentiment_counts[sentiment] += 1
        results.append({
            "tweet": tweet,
            "sentiment": sentiment,
            "score": round(confidence, 3)
        })

    # Determine overall sentiment (by count)
    overall = max(sentiment_counts, key=sentiment_counts.get)

    # Sort for top positives/negatives by confidence
    positives = sorted(
        (r for r in results if r["sentiment"] == "positive"),
        key=lambda x: x["score"],
        reverse=True
    )[:3]
    negatives = sorted(
        (r for r in results if r["sentiment"] == "negative"),
        key=lambda x: x["score"],
        reverse=True
    )[:3]

    return jsonify({
        "counts": sentiment_counts,
        "overall": overall,
        "top_positive": positives,
        "top_negative": negatives
    })


if __name__ == "__main__":
    app.run(debug=True)
