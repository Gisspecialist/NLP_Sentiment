import io
from datetime import datetime
import json
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Make sure VADER lexicon is available
nltk.download("vader_lexicon", quiet=True)


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------

def guess_column(columns, candidates):
    """
    Try to guess a column name from a list of candidates.
    Returns first match or None.
    """
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = " ".join(text.split())
    return text.lower()


@st.cache_data(show_spinner=False)
def compute_sentiment(clean_text_series: pd.Series) -> pd.DataFrame:
    sia = SentimentIntensityAnalyzer()
    scores = clean_text_series.apply(lambda t: sia.polarity_scores(t) if t else {"compound": 0.0})
    df_sent = pd.DataFrame(scores.tolist())
    df_sent.rename(columns={"compound": "sentiment_compound"}, inplace=True)
    df_sent["sentiment_signed"] = df_sent["sentiment_compound"]
    df_sent["sentiment_label"] = df_sent["sentiment_compound"].apply(
        lambda c: "POS" if c > 0.05 else ("NEG" if c < -0.05 else "NEU")
    )
    return df_sent[["sentiment_compound", "sentiment_signed", "sentiment_label"]]


@st.cache_data(show_spinner=False)
def compute_topics(texts, max_features=3000, svd_components=100, n_clusters=6):
    """
    Build TF-IDF + SVD embeddings and cluster into topics with K-Means.
    Returns: topic_ids (array-like), embeddings (2D array).
    Falls back gracefully for very small datasets.
    """
    # If too few docs, no clustering
    if len(texts) < 3:
        return np.zeros(len(texts), dtype=int), None

    # Try bigram TF-IDF, min_df=2; fall back if nothing survives
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
        )
        X_tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            min_df=1,
        )
        X_tfidf = vectorizer.fit_transform(texts)

    if X_tfidf.shape[1] <= 1:
        # not enough terms for meaningful clustering
        return np.zeros(len(texts), dtype=int), None

    n_comp = min(svd_components, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp)
    X_emb = svd.fit_transform(X_tfidf)

    n_samples = X_emb.shape[0]
    k = min(max(2, n_clusters), n_samples)  # at least 2, at most n_samples

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(X_emb)

    if k == 1:
        topic_ids = np.zeros(n_samples, dtype=int)
    else:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        topic_ids = kmeans.fit_predict(emb_scaled)

    return topic_ids, X_emb


def sentiment_over_time(df, ts_col="timestamp", freq="D"):
    d = df.set_index(ts_col)
    agg = (
        d.resample(freq)
         .agg(
             mean_sentiment=("sentiment_signed", "mean"),
             volume=("sentiment_signed", "count"),
         )
         .dropna()
         .reset_index()
    )
    return agg


# --------------------------------------------------------------------------------
def build_standalone_html(df):
    """
    Build a standalone HTML dashboard with the current (filtered) data baked in.
    Uses Plotly + vanilla JS; no Python or Streamlit needed to run it.
    """
    # We only need a subset of columns; ensure they exist
    cols_needed = ["timestamp", "sentiment_signed", "program", "location"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for HTML export: {missing}")

    # Convert timestamps to ISO strings (no timezone objects)
    df_export = df.copy()
    df_export["timestamp"] = pd.to_datetime(df_export["timestamp"], errors="coerce")
    df_export = df_export.dropna(subset=["timestamp"])
    df_export["timestamp"] = df_export["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    records = df_export[cols_needed].to_dict(orient="records")
    data_json = json.dumps(records)  # will be injected into JS as DATA constant

    # HTML template; __DATA__ will be replaced with JSON
    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Narrative Intelligence Standalone Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                   Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      background: #f5f7fb;
      color: #222;
    }
    header {
      background: #0b3c5d;
      color: #fff;
      padding: 16px 24px;
    }
    header h1 {
      margin: 0 0 4px;
      font-size: 24px;
    }
    header p {
      margin: 0;
      opacity: 0.85;
      font-size: 14px;
    }
    main {
      max-width: 1200px;
      margin: 20px auto 40px;
      padding: 0 16px;
    }
    section {
      background: #fff;
      border-radius: 8px;
      padding: 16px 18px 18px;
      margin-bottom: 16px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    h2 {
      margin-top: 0;
      border-bottom: 2px solid #e2e6f0;
      padding-bottom: 4px;
      font-size: 18px;
    }
    label {
      font-size: 13px;
      font-weight: 600;
      display: block;
      margin-bottom: 3px;
    }
    input[type="date"], select {
      font-size: 13px;
      padding: 4px 6px;
      border-radius: 4px;
      border: 1px solid #cbd5e1;
      width: 100%;
      box-sizing: border-box;
    }
    .row { display: flex; flex-wrap: wrap; gap: 10px; }
    .col-3 { flex: 1 1 220px; }
    button {
      background: #0b3c5d;
      color: #fff;
      border: none;
      border-radius: 999px;
      padding: 6px 14px;
      font-size: 13px;
      cursor: pointer;
    }
    .small { font-size: 12px; color: #555; margin-top: 4px; }
    .chart {
      margin-top: 10px;
      border-radius: 8px;
      background: #f8fafc;
      border: 1px solid #e2e6f0;
      padding: 8px;
    }
    .chart-title {
      font-size: 14px;
      font-weight: 600;
      margin: 0 0 4px;
    }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #e5f3ff;
      color: #245a8d;
      font-size: 11px;
      margin-left: 4px;
    }
  </style>
</head>
<body>
<header>
  <h1>Narrative Intelligence Standalone Dashboard</h1>
  <p>Filters &amp; charts generated from an exported dataset – no Python server required.</p>
</header>

<main>
  <section>
    <h2>1. Filters</h2>
    <div class="row">
      <div class="col-3">
        <label for="startDate">Start date</label>
        <input type="date" id="startDate" />
      </div>
      <div class="col-3">
        <label for="endDate">End date</label>
        <input type="date" id="endDate" />
      </div>
      <div class="col-3">
        <label for="programFilter">Program(s)</label>
        <select id="programFilter" multiple size="4"></select>
        <div class="small">Hold Ctrl (Cmd on Mac) to select multiple.</div>
      </div>
      <div class="col-3">
        <label for="locationFilter">Location(s)</label>
        <select id="locationFilter" multiple size="4"></select>
        <div class="small">Leave empty to include all.</div>
      </div>
    </div>
    <div style="margin-top:8px;">
      <button id="applyFiltersBtn">Apply filters</button>
      <span id="filterInfo" class="small"></span>
    </div>
  </section>

  <section>
    <h2>2. Charts</h2>
    <div class="chart">
      <p class="chart-title">Sentiment over Time<span class="pill">Time &amp; Volume</span></p>
      <div id="chartTime" style="height:320px;"></div>
      <p class="small">Line = average sentiment; bars = volume.</p>
    </div>
    <div class="chart">
      <p class="chart-title">Average Sentiment by Program<span class="pill">Program</span></p>
      <div id="chartProg" style="height:320px;"></div>
    </div>
    <div class="chart">
      <p class="chart-title">Average Sentiment by Location<span class="pill">Location</span></p>
      <div id="chartLoc" style="height:320px;"></div>
    </div>
  </section>
</main>

<script>
  // Data baked in by Streamlit (Python)
  const DATA = __DATA__;

  let workingRows = [];
  let filteredRows = [];
  let dateRange = { min: null, max: null };

  function initData() {
    workingRows = DATA.map(r => {
      const ts = new Date(r.timestamp);
      const d = isNaN(ts.getTime()) ? null : ts;
      const dateOnly = d ? d.toISOString().slice(0, 10) : null;
      return {
        ts: d,
        dateOnly: dateOnly,
        sentiment: parseFloat(r.sentiment_signed || 0),
        program: String(r.program || "unspecified"),
        location: String(r.location || "unspecified")
      };
    }).filter(r => r.ts && r.dateOnly);
  }

  function initFilters() {
    if (!workingRows.length) return;
    const dates = workingRows.map(r => r.dateOnly).sort();
    dateRange.min = dates[0];
    dateRange.max = dates[dates.length - 1];

    const startInput = document.getElementById("startDate");
    const endInput   = document.getElementById("endDate");
    startInput.min = dateRange.min;
    startInput.max = dateRange.max;
    endInput.min   = dateRange.min;
    endInput.max   = dateRange.max;
    startInput.value = dateRange.min;
    endInput.value   = dateRange.max;

    // programs
    const progSel = document.getElementById("programFilter");
    progSel.innerHTML = "";
    const programs = Array.from(new Set(workingRows.map(r => r.program))).sort();
    programs.forEach(p => {
      const opt = document.createElement("option");
      opt.value = p;
      opt.textContent = p;
      progSel.appendChild(opt);
    });

    // locations
    const locSel = document.getElementById("locationFilter");
    locSel.innerHTML = "";
    const locs = Array.from(new Set(workingRows.map(r => r.location))).sort();
    locs.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l;
      opt.textContent = l;
      locSel.appendChild(opt);
    });

    document.getElementById("filterInfo").textContent =
      "Using date range " + dateRange.min + " → " + dateRange.max + ".";
  }

  function getSelectedValues(selectEl) {
    const vals = [];
    Array.from(selectEl.options).forEach(opt => {
      if (opt.selected) vals.push(opt.value);
    });
    return vals;
  }

  function applyFilters() {
    if (!workingRows.length) return;

    const startVal = document.getElementById("startDate").value || dateRange.min;
    const endVal   = document.getElementById("endDate").value   || dateRange.max;

    const progSel = getSelectedValues(document.getElementById("programFilter"));
    const locSel  = getSelectedValues(document.getElementById("locationFilter"));

    filteredRows = workingRows.filter(r => {
      if (r.dateOnly < startVal || r.dateOnly > endVal) return false;
      if (progSel.length > 0 && !progSel.includes(r.program)) return false;
      if (locSel.length > 0 && !locSel.includes(r.location)) return false;
      return true;
    });

    document.getElementById("filterInfo").textContent =
      "Filtered to " + filteredRows.length + " messages.";
    drawAllCharts();
  }

  // --- chart helpers ---
  function groupByDate(rows) {
    const map = {};
    rows.forEach(r => {
      if (!map[r.dateOnly]) map[r.dateOnly] = { sum: 0, count: 0 };
      map[r.dateOnly].sum += r.sentiment;
      map[r.dateOnly].count += 1;
    });
    const keys = Object.keys(map).sort();
    const dates = [], avgSent = [], vols = [];
    keys.forEach(k => {
      dates.push(k);
      avgSent.push(map[k].sum / map[k].count);
      vols.push(map[k].count);
    });
    return { dates, avgSent, vols };
  }

  function groupByKey(rows, key) {
    const map = {};
    rows.forEach(r => {
      const k = r[key] || "unspecified";
      if (!map[k]) map[k] = { sum: 0, count: 0 };
      map[k].sum += r.sentiment;
      map[k].count += 1;
    });
    const keys = Object.keys(map).sort();
    const avg = [], vol = [];
    keys.forEach(k => {
      avg.push(map[k].sum / map[k].count);
      vol.push(map[k].count);
    });
    return { keys, avg, vol };
  }

  function drawSentimentOverTime() {
    const div = document.getElementById("chartTime");
    if (!filteredRows.length) {
      Plotly.newPlot(div, [], { title: "No data after filters" }, { displayModeBar: false });
      return;
    }
    const g = groupByDate(filteredRows);
    const trace1 = {
      x: g.dates,
      y: g.avgSent,
      type: "scatter",
      mode: "lines+markers",
      name: "Mean sentiment",
      yaxis: "y1"
    };
    const trace2 = {
      x: g.dates,
      y: g.vols,
      type: "bar",
      name: "Volume",
      yaxis: "y2",
      opacity: 0.4
    };
    const layout = {
      margin: { t: 10, r: 40, b: 40, l: 40 },
      yaxis: { title: "Sentiment", range: [-1, 1] },
      yaxis2: { title: "Volume", overlaying: "y", side: "right" },
      showlegend: true
    };
    Plotly.newPlot(div, [trace1, trace2], layout, { responsive: true, displayModeBar: false });
  }

  function drawSentimentByProgram() {
    const div = document.getElementById("chartProg");
    if (!filteredRows.length) {
      Plotly.newPlot(div, [], { title: "No data" }, { displayModeBar: false });
      return;
    }
    const g = groupByKey(filteredRows, "program");
    const data = [{
      x: g.avg,
      y: g.keys,
      type: "bar",
      orientation: "h"
    }];
    const layout = {
      margin: { t: 10, r: 20, b: 40, l: 140 },
      xaxis: { title: "Average sentiment", range: [-1, 1] },
      showlegend: false
    };
    Plotly.newPlot(div, data, layout, { responsive: true, displayModeBar: false });
  }

  function drawSentimentByLocation() {
    const div = document.getElementById("chartLoc");
    if (!filteredRows.length) {
      Plotly.newPlot(div, [], { title: "No data" }, { displayModeBar: false });
      return;
    }
    const g = groupByKey(filteredRows, "location");
    const data = [{
      x: g.avg,
      y: g.keys,
      type: "bar",
      orientation: "h"
    }];
    const layout = {
      margin: { t: 10, r: 20, b: 40, l: 140 },
      xaxis: { title: "Average sentiment", range: [-1, 1] },
      showlegend: false
    };
    Plotly.newPlot(div, data, layout, { responsive: true, displayModeBar: false });
  }

  function drawAllCharts() {
    drawSentimentOverTime();
    drawSentimentByProgram();
    drawSentimentByLocation();
  }

  // init
  window.addEventListener("load", () => {
    initData();
    initFilters();
    applyFilters();
    document.getElementById("applyFiltersBtn").addEventListener("click", applyFilters);
  });
</script>
</body>
</html>
"""
    html = template.replace("__DATA__", data_json)
    return html

# --------------------------------------------------------------------------------

st.set_page_config(
    page_title="Narrative Intelligence Dashboard",
    layout="wide",
)

st.title("Narrative Intelligence Dashboard")
st.markdown(
    "Upload any CSV with text, timestamps, and optional location/program/project columns "
    "to explore sentiment, topics, and indicators interactively."
)

# -------------------
# File upload
# -------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("👆 Upload a CSV to begin. At minimum you need a timestamp and text column.")
    st.stop()

# Load data
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

if raw_df.empty:
    st.warning("The uploaded CSV is empty.")
    st.stop()

st.subheader("Step 1 – Map Columns")

cols = list(raw_df.columns)

# Try to guess typical columns
guess_ts = guess_column(cols, ["timestamp", "time", "date"])
guess_text = guess_column(cols, ["text", "message", "content", "body"])
guess_source = guess_column(cols, ["source", "actor", "channel"])
guess_location = guess_column(cols, ["location", "city", "district", "area"])
guess_program = guess_column(cols, ["program", "programme", "project_component"])
guess_project = guess_column(cols, ["project_id", "project", "proj_id"])

col1, col2, col3 = st.columns(3)

with col1:
    ts_col = st.selectbox("Timestamp column", options=["<none>"] + cols, index=(cols.index(guess_ts) + 1) if guess_ts else 0)
with col2:
    text_col = st.selectbox("Text column", options=["<none>"] + cols, index=(cols.index(guess_text) + 1) if guess_text else 0)
with col3:
    source_col = st.selectbox("Source / Actor column (optional)", options=["<none>"] + cols,
                              index=(cols.index(guess_source) + 1) if guess_source else 0)

col4, col5, col6 = st.columns(3)
with col4:
    location_col = st.selectbox("Location column (optional)", options=["<none>"] + cols,
                                index=(cols.index(guess_location) + 1) if guess_location else 0)
with col5:
    program_col = st.selectbox("Program column (optional)", options=["<none>"] + cols,
                               index=(cols.index(guess_program) + 1) if guess_program else 0)
with col6:
    project_col = st.selectbox("Project ID column (optional)", options=["<none>"] + cols,
                               index=(cols.index(guess_project) + 1) if guess_project else 0)

if ts_col == "<none>" or text_col == "<none>":
    st.error("You must at least select a timestamp column and a text column.")
    st.stop()

# -------------------
# Build working DataFrame
# -------------------
df = raw_df.copy()

# Parse timestamp
df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp"])

# Canonical text column
df["text"] = df[text_col].astype(str)
df["clean_text"] = df["text"].apply(basic_clean)

# Optional canonical columns
if source_col != "<none>":
    df["source"] = df[source_col].astype(str)
else:
    df["source"] = "unknown"

if location_col != "<none>":
    df["location"] = df[location_col].astype(str)
else:
    df["location"] = "unspecified"

if program_col != "<none>":
    df["program"] = df[program_col].astype(str)
else:
    df["program"] = "unspecified"

if project_col != "<none>":
    df["project_id"] = df[project_col].astype(str)
else:
    df["project_id"] = "unspecified"

# Assign an ID if none exists
if "id" not in df.columns:
    df["id"] = np.arange(1, len(df) + 1)

st.write(f"Loaded **{len(df)}** messages after parsing timestamps.")

# -------------------
# Sentiment & Topics
# -------------------
st.subheader("Step 2 – Enrich with Sentiment & Topics")

# Sentiment
if not {"sentiment_signed", "sentiment_label"}.issubset(df.columns):
    with st.spinner("Computing sentiment (VADER)..."):
        sent_df = compute_sentiment(df["clean_text"])
        df = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)
    st.success("Sentiment computed and added.")
else:
    st.info("Using existing sentiment columns from the CSV.")

# Topics
if "topic_id" not in df.columns:
    n_clusters = st.slider("Number of topics (clusters)", min_value=3, max_value=12, value=6)
    with st.spinner("Computing topics from text (TF-IDF + SVD + K-Means)..."):
        topic_ids, _emb = compute_topics(df["clean_text"].tolist(), n_clusters=n_clusters)
        df["topic_id"] = topic_ids
    st.success("Topics assigned.")
else:
    st.info("Using existing topic_id column from the CSV.")

# Time bin for later (month-level by default)
df["time_bin"] = df["timestamp"].dt.to_period("M").astype(str)

# -------------------
# Filters
# -------------------
st.subheader("Step 3 – Filter Data")

col_date1, col_date2 = st.columns(2)
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

with col_date1:
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
with col_date2:
    end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

mask_time = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)

program_options = sorted(df["program"].unique())
location_options = sorted(df["location"].unique())

col_f1, col_f2 = st.columns(2)
with col_f1:
    selected_programs = st.multiselect("Filter by program", options=program_options, default=program_options)
with col_f2:
    selected_locations = st.multiselect("Filter by location", options=location_options, default=location_options)

mask_program = df["program"].isin(selected_programs)
mask_location = df["location"].isin(selected_locations)

df_f = df[mask_time & mask_program & mask_location].copy()

st.write(f"Filtered down to **{len(df_f)}** messages.")

if df_f.empty:
    st.warning("No data after filters. Adjust filters or date range.")
    st.stop()

# -------------------
# Tabs for dashboard
# -------------------
tab_time, tab_program, tab_location, tab_topics, tab_download = st.tabs(
    ["Time & Sentiment", "Programs", "Locations", "Topics", "Download"]
)

# -------------------
# TIME & SENTIMENT TAB
# -------------------
with tab_time:
    st.header("Time & Sentiment")

    sent_trend = sentiment_over_time(df_f, ts_col="timestamp", freq="D")
    if sent_trend.empty:
        st.info("Not enough data points to plot time series.")
    else:
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(sent_trend["timestamp"], sent_trend["mean_sentiment"], marker="o")
        ax1.set_ylabel("Mean Sentiment")
        ax1.set_xlabel("Date")
        ax1.set_title("Sentiment Over Time")

        ax2 = ax1.twinx()
        ax2.bar(sent_trend["timestamp"], sent_trend["volume"], alpha=0.3)
        ax2.set_ylabel("Message Volume")

        fig.tight_layout()
        st.pyplot(fig)

    st.markdown(
        "- **Line** shows average sentiment (−1 to +1).  \n"
        "- **Bars** show message volume.  \n"
        "Spikes + negative sentiment flag emerging concerns; spikes + positive sentiment "
        "often signal successes or good communication."
    )

# -------------------
# PROGRAM TAB
# -------------------
with tab_program:
    st.header("Program View")

    prog_sent = (
        df_f.groupby("program")
           .agg(
               avg_sentiment=("sentiment_signed", "mean"),
               volume=("id", "count"),
           )
           .sort_values("avg_sentiment", ascending=True)
    )

    if prog_sent.empty:
        st.info("No programme information available after filtering.")
    else:
        st.subheader("Average Sentiment by Program")
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.barh(prog_sent.index, prog_sent["avg_sentiment"])
        ax1.axvline(0, linewidth=0.8)
        ax1.set_xlabel("Average Sentiment (compound)")
        fig1.tight_layout()
        st.pyplot(fig1)

        st.subheader("Program Volume vs Sentiment")
        prog_sent_reset = prog_sent.reset_index()
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.scatter(prog_sent_reset["volume"], prog_sent_reset["avg_sentiment"])
        for _, row in prog_sent_reset.iterrows():
            ax2.text(row["volume"], row["avg_sentiment"], row["program"], fontsize=8)
        ax2.set_xlabel("Message Volume")
        ax2.set_ylabel("Average Sentiment")
        fig2.tight_layout()
        st.pyplot(fig2)

        st.markdown(
            "- Programmes at the right/top = many messages with positive sentiment.  \n"
            "- Left/bottom = lower sentiment and/or low engagement (potential red flags)."
        )

# -------------------
# LOCATION TAB
# -------------------
with tab_location:
    st.header("Location View")

    loc_sent = (
        df_f.groupby("location")
           .agg(
               avg_sentiment=("sentiment_signed", "mean"),
               volume=("id", "count"),
           )
           .sort_values("avg_sentiment", ascending=True)
    )

    if loc_sent.empty:
        st.info("No location information available after filtering.")
    else:
        st.subheader("Average Sentiment by Location")
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.barh(loc_sent.index, loc_sent["avg_sentiment"])
        ax1.axvline(0, linewidth=0.8)
        ax1.set_xlabel("Average Sentiment (compound)")
        fig1.tight_layout()
        st.pyplot(fig1)

        st.markdown(
            "Locations with lower scores and sufficient volume can be flagged for deeper "
            "qualitative follow-up or targeted interventions."
        )

# -------------------
# TOPICS TAB
# -------------------
with tab_topics:
    st.header("Topic Mix – by Program & Location")

    # Topic by program
    if "topic_id" in df_f.columns:
        topic_prog = (
            df_f.groupby(["program", "topic_id"])
               .size()
               .reset_index(name="count")
        )
        heat_prog = topic_prog.pivot(index="program", columns="topic_id", values="count").fillna(0)

        if not heat_prog.empty:
            st.subheader("Topic Volume by Program")
            fig_prog, ax_prog = plt.subplots(figsize=(8, 5))
            im1 = ax_prog.imshow(heat_prog.values, aspect="auto")

            ax_prog.set_yticks(range(heat_prog.shape[0]))
            ax_prog.set_yticklabels(heat_prog.index)
            ax_prog.set_xticks(range(heat_prog.shape[1]))
            ax_prog.set_xticklabels(heat_prog.columns)

            ax_prog.set_xlabel("Topic ID")
            ax_prog.set_ylabel("Program")
            cbar1 = plt.colorbar(im1, ax=ax_prog)
            cbar1.set_label("Message Count")
            fig_prog.tight_layout()
            st.pyplot(fig_prog)
        else:
            st.info("Not enough data to compute topic-by-program heatmap.")

        # Topic by location
        topic_loc = (
            df_f.groupby(["location", "topic_id"])
               .size()
               .reset_index(name="count")
        )
        heat_loc = topic_loc.pivot(index="location", columns="topic_id", values="count").fillna(0)

        if not heat_loc.empty:
            st.subheader("Topic Volume by Location")
            fig_loc, ax_loc = plt.subplots(figsize=(8, 5))
            im2 = ax_loc.imshow(heat_loc.values, aspect="auto")

            ax_loc.set_yticks(range(heat_loc.shape[0]))
            ax_loc.set_yticklabels(heat_loc.index)
            ax_loc.set_xticks(range(heat_loc.shape[1]))
            ax_loc.set_xticklabels(heat_loc.columns)

            ax_loc.set_xlabel("Topic ID")
            ax_loc.set_ylabel("Location")
            cbar2 = plt.colorbar(im2, ax=ax_loc)
            cbar2.set_label("Message Count")
            fig_loc.tight_layout()
            st.pyplot(fig_loc)
        else:
            st.info("Not enough data to compute topic-by-location heatmap.")
    else:
        st.info("No topic_id column available; cannot build topic heatmaps.")

    st.markdown(
        "These heatmaps reveal **which themes are tied to which programmes and locations**, "
        "helping identify where specific issues (topics) are concentrated."
    )

# -------------------
# DOWNLOAD TAB
# -------------------
with tab_download:
    st.header("Download Enriched Data")

    st.write(
        "This file includes the original columns plus "
        "<code>clean_text</code>, <code>sentiment_compound</code>, "
        "<code>sentiment_signed</code>, <code>sentiment_label</code>, "
        "<code>topic_id</code>, and <code>time_bin</code> for use in other tools "
        "(e.g. your knowledge graph builder).",
        unsafe_allow_html=True
    )

    # We give the filtered + enriched data so it matches what the user is seeing.
    out_buf = io.StringIO()
    df_f.to_csv(out_buf, index=False)
    data_bytes = out_buf.getvalue().encode("utf-8")

    st.download_button(
        label="Download enriched CSV",
        data=data_bytes,
        file_name="messages_enriched_dashboard.csv",
        mime="text/csv",
    )

    st.write("You can feed this enriched CSV into your existing graph/Quarto pipelines.")
