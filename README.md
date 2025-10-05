# OpinionTally: Twitter Sentiment Analysis on Vaccination Opinions

Flask web app that performs sentiment analysis on vaccination-related tweets/opinions using **NLTK VADER**. Frontend uses **Bootstrap 5** and **Chart.js**. All history is stored locally in the browser's `localStorage` (no database, no authentication).

## Features
* **Analysis:** Paste multiple tweets/opinions (one per line) on the Home page.
* **Sentiment Engine:** Uses VADER (Valence Aware Dictionary and sEntiment Reasoner) for accurate social media sentiment scoring.
* **Visualization:** Displays a responsive Chart.js chart showing the overall sentiment distribution (Positive / Neutral / Negative).
* **Summary:** Shows the overall compound score, sentiment counts, and identifies the **top 3 most positive** and **top 3 most negative** opinions.
* **Local History:** The most recent analysis result is **automatically saved to browser `localStorage` upon page refresh** or successful completion of a new analysis.
* **History Page:** Lists all past analyses stored in your browser, including summary details and a link to view input and scores.

---

## Setup & Run (VS Code / Local Environment)

**Prerequisites:** Python 3.7+ and `pip`.

1.  **Clone or Copy the Project Folder:**
    ```bash
    cd twitter-vax-sentiment
    ```

2.  **Create and Activate Python Virtual Environment:**

    **macOS / Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

    **Windows (Command Prompt/Git Bash):**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    
    *(Note: If using PowerShell on Windows, you may need `venv\Scripts\Activate.ps1`)*

3.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
    *(The application will automatically download the necessary NLTK VADER data on its first run.)*

4.  **Run the Flask App:**
    ```bash
    python app.py
    ```

5.  **Open your browser:**
    Navigate to `http://127.0.0.1:5000/`.
