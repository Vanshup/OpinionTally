
from flask import Flask, render_template, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Ensure VADER lexicon is present; download if missing
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    # This will download the lexicon to NLTK data directory
    nltk.download("vader_lexicon")

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()


@app.route("/")
def home():
    """Render Home page (input / analyze UI)."""
    return render_template("home.html")


@app.route("/history")
def history():
    """Render History page (reads from browser localStorage)."""
    return render_template("history.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze route: expects JSON { "text": "<multiple lines>" }
    Each non-empty line is treated as a single tweet/opinion.
    Returns JSON with counts, overall, top_positive/top_negative lists.
    """
    data = request.get_json(force=True, silent=True)
    text_input = ""
    if data:
        text_input = data.get("text", "") or ""

    text_input = text_input.strip()
    if not text_input:
        return jsonify({"error": "Input is empty. Please enter some text."}), 400

    # Split by newline and filter out empty lines
    tweets = [line.strip() for line in text_input.splitlines() if line.strip()]

    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    results = []

    for tweet in tweets:
        # Get VADER scores: {'neg':..., 'neu':..., 'pos':..., 'compound':...}
        scores = sia.polarity_scores(tweet)
        compound = scores["compound"]

        # Classic thresholds for VADER
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        sentiment_counts[sentiment] += 1
        results.append({"tweet": tweet, "sentiment": sentiment, "score": compound})

    # Compute overall sentiment by majority count (ties resolved by compound sums)
    # Basic overall: the sentiment with highest count
    overall = max(sentiment_counts, key=sentiment_counts.get)

    # Build Top Positive and Top Negative lists (no duplicates).
    # Sort positives by score descending, negatives by score ascending (more negative first).
    positives = sorted(
        (r for r in results if r["sentiment"] == "positive"),
        key=lambda x: x["score"],
        reverse=True
    )[:3]

    negatives = sorted(
        (r for r in results if r["sentiment"] == "negative"),
        key=lambda x: x["score"]
    )[:3]

    # Return structured JSON to frontend
    return jsonify({
        "counts": sentiment_counts,
        "overall": overall,
        "top_positive": positives,
        "top_negative": negatives
    })


if __name__ == "__main__":
    # Debug True for development. Turn off in production.
    app.run(debug=True)