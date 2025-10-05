/* Globals to hold chart instances so we can destroy before re-rendering */
let homeChart = null;
const historyCharts = {}; // map idx -> Chart instance

document.addEventListener("DOMContentLoaded", () => {
  const analyzeBtn = document.getElementById("analyzeBtn");
  const clearBtn = document.getElementById("clearBtn");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  if (analyzeBtn) analyzeBtn.addEventListener("click", runAnalysis);
  if (clearBtn) clearBtn.addEventListener("click", () => {
    document.getElementById("tweetInput").value = "";
    document.getElementById("resultSection").style.display = "none";
  });
  if (clearHistoryBtn) clearHistoryBtn.addEventListener("click", clearHistory);

  // If on Home page and lastResult exists, restore it to the UI (but do not push to history yet)
  const lastResultRaw = localStorage.getItem("lastResult");
  if (lastResultRaw && document.getElementById("resultSection")) {
    try {
      const last = JSON.parse(lastResultRaw);
      if (last && last.data) {
        displayResults(last.data);
      }
    } catch (e) {
      console.warn("Failed to parse lastResult:", e);
    }
  }

  // If on History page, render history and move lastResult into history if present
  if (document.getElementById("historyContainer")) {
    // loadHistory handles moving lastResult -> history
    loadHistory();
  }
});

/* Send text to backend analyze endpoint */
async function runAnalysis() {
  const textarea = document.getElementById("tweetInput");
  const text = (textarea.value || "").trim();

  if (!text) {
    alert("Please enter tweets/opinions (one per line) before analyzing.");
    return;
  }

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      const err = await response.json();
      alert(err.error || "Error analyzing text.");
      return;
    }

    const data = await response.json();
    // Show results on screen
    displayResults(data);

    // Save lastResult to localStorage (not pushing to history yet).
    const lastPayload = {
      timestamp: new Date().toLocaleString(),
      data: data
    };
    localStorage.setItem("lastResult", JSON.stringify(lastPayload));

    // Note: the requirement was to save to history only upon refresh.
    // We still save lastResult so visiting History or refreshing will push it into history.
  } catch (err) {
    console.error("Analyze error:", err);
    alert("Unexpected error while analyzing. See console for details.");
  }
}

/* Render results on Home page, including chart and top lists */
function displayResults(data) {
  const resultSection = document.getElementById("resultSection");
  resultSection.style.display = "block";

  const summary = document.getElementById("summaryText");
  summary.innerText = `Overall Sentiment: ${String(data.overall).toUpperCase()}`;

  // Destroy existing home chart if present
  if (homeChart) {
    try { homeChart.destroy(); } catch (e) { /* ignore */ }
    homeChart = null;
  }

  // Create chart in the container. maintainAspectRatio=false lets CSS height control size.
  const ctx = document.getElementById("sentimentChart").getContext("2d");
  homeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Positive", "Neutral", "Negative"],
      datasets: [{
        data: [data.counts.positive, data.counts.neutral, data.counts.negative],
        backgroundColor: ["#20c997", "#ffc107", "#dc3545"], // teal, amber, crimson
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } }
      }
    }
  });

  // Populate Top Positive and Negative lists (only actual items)
  const posList = document.getElementById("positiveList");
  const negList = document.getElementById("negativeList");
  posList.innerHTML = "";
  negList.innerHTML = "";

  if (Array.isArray(data.top_positive) && data.top_positive.length > 0) {
    data.top_positive.forEach(item => {
      const li = document.createElement("li");
      li.className = "list-group-item";
      li.innerHTML = sanitize(item.tweet);
      posList.appendChild(li);
    });
  } else {
    posList.innerHTML = `<li class="list-group-item text-muted">No positive tweets found.</li>`;
  }

  if (Array.isArray(data.top_negative) && data.top_negative.length > 0) {
    data.top_negative.forEach(item => {
      const li = document.createElement("li");
      li.className = "list-group-item";
      li.innerHTML = sanitize(item.tweet);
      negList.appendChild(li);
    });
  } else {
    negList.innerHTML = `<li class="list-group-item text-muted">No negative tweets found.</li>`;
  }
}

/* Move lastResult (if any) into history and render the accordion history */
function loadHistory() {
  const container = document.getElementById("historyContainer");

  // If there's a lastResult, push it onto history (this simulates "save on refresh")
  const lastRaw = localStorage.getItem("lastResult");
  if (lastRaw) {
    try {
      const lastObj = JSON.parse(lastRaw);
      if (lastObj) {
        const historyArray = JSON.parse(localStorage.getItem("history") || "[]");
        historyArray.push(lastObj);
        localStorage.setItem("history", JSON.stringify(historyArray));
      }
      // Remove lastResult once moved to history
      localStorage.removeItem("lastResult");
    } catch (e) {
      console.warn("Failed to move lastResult to history:", e);
    }
  }

  const history = JSON.parse(localStorage.getItem("history") || "[]");

  if (!history || history.length === 0) {
    container.innerHTML = `<div class="card p-3 text-muted">No history yet. Run an analysis and refresh to save it here.</div>`;
    return;
  }

  // Build accordion
  container.innerHTML = "";
  history.forEach((entry, idx) => {
    const itemId = `collapse${idx}`;
    const headingId = `heading${idx}`;

    // Create accordion item
    const accordionItem = document.createElement("div");
    accordionItem.className = "accordion-item mb-2";
    accordionItem.innerHTML = `
      <h2 class="accordion-header" id="${headingId}">
        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#${itemId}" aria-expanded="false" aria-controls="${itemId}">
          ${sanitize(entry.timestamp)} — Overall: ${String(entry.data.overall).toUpperCase()}
        </button>
      </h2>
      <div id="${itemId}" class="accordion-collapse collapse" aria-labelledby="${headingId}" data-bs-parent="#historyContainer">
        <div class="accordion-body">
          <p><strong>Counts:</strong> ${entry.data.counts.positive} Positive, ${entry.data.counts.neutral} Neutral, ${entry.data.counts.negative} Negative</p>
          <div class="chart-sm mb-2"><canvas id="chart_${idx}" style="width:100%; height:100%;"></canvas></div>
          <div class="row">
            <div class="col-12 col-md-6">
              <h6>Top Positive</h6>
              <ul id="poslist_${idx}" class="list-group mb-2"></ul>
            </div>
            <div class="col-12 col-md-6">
              <h6>Top Negative</h6>
              <ul id="neglist_${idx}" class="list-group mb-2"></ul>
            </div>
          </div>
        </div>
      </div>
    `;
    container.appendChild(accordionItem);

    // Populate top lists safely
    const posList = document.getElementById(`poslist_${idx}`);
    const negList = document.getElementById(`neglist_${idx}`);

    if (Array.isArray(entry.data.top_positive) && entry.data.top_positive.length > 0) {
      entry.data.top_positive.forEach(t => {
        const li = document.createElement("li");
        li.className = "list-group-item";
        li.innerHTML = sanitize(t.tweet);
        posList.appendChild(li);
      });
    } else {
      posList.innerHTML = `<li class="list-group-item text-muted">No positive tweets.</li>`;
    }

    if (Array.isArray(entry.data.top_negative) && entry.data.top_negative.length > 0) {
      entry.data.top_negative.forEach(t => {
        const li = document.createElement("li");
        li.className = "list-group-item";
        li.innerHTML = sanitize(t.tweet);
        negList.appendChild(li);
      });
    } else {
      negList.innerHTML = `<li class="list-group-item text-muted">No negative tweets.</li>`;
    }

    // Defer chart creation slightly so the canvas is available in DOM & collapsed content size is meaningful
    setTimeout(() => {
      const ctx = document.getElementById(`chart_${idx}`);
      if (!ctx) return;

      // Destroy previous chart if exists
      if (historyCharts[idx]) {
        try { historyCharts[idx].destroy(); } catch (e) { /* ignore */ }
        historyCharts[idx] = null;
      }

      historyCharts[idx] = new Chart(ctx.getContext("2d"), {
        type: "bar",
        data: {
          labels: ["Positive", "Neutral", "Negative"],
          datasets: [{
            data: [
              entry.data.counts.positive,
              entry.data.counts.neutral,
              entry.data.counts.negative
            ],
            backgroundColor: ["#20c997", "#ffc107", "#dc3545"],
            borderRadius: 4
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
        }
      });
    }, 80);
  });
}

/* Clear history (localStorage + re-render) */
function clearHistory() {
  if (!confirm("Clear all saved history? This cannot be undone.")) return;
  localStorage.removeItem("history");
  localStorage.removeItem("lastResult");
  // Re-render history if on page
  const container = document.getElementById("historyContainer");
  if (container) container.innerHTML = `<div class="card p-3 text-muted">No history yet. Run an analysis and refresh to save it here.</div>`;
}

/* Sanitize text to avoid XSS when displaying tweets */
function sanitize(text) {
  const d = document.createElement("div");
  d.innerText = text;
  return d.innerHTML;
}
