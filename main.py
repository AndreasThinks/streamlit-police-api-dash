"""
UK Police Priorities topic explorer optimised for Streamlit Cloud.

Key points:
1. Downloads data directly from Kaggle when credentials are supplied via Streamlit secrets.
2. Ships with a tiny sample dataset as a fallback for quick previews.
3. Uses lightweight TF-IDF embeddings (no torch) to keep resource usage low.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from bertopic import BERTopic
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Global constants
APP_ROOT = Path(__file__).parent
DEFAULT_DATA_DIR = APP_ROOT / "data" / "sample"
pio.templates.default = "plotly_dark"

st.set_page_config(page_title="UK Police Topics Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .st-cx { color: #e0e0e0 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------
def _locate_dataset_folder(base_dir: str) -> str:
    """Find the folder that actually holds the CSV files."""
    base = Path(base_dir)
    direct = base / "priorities.csv"
    if direct.exists():
        return str(base)
    matches = sorted(base.rglob("priorities.csv"))
    if matches:
        return str(matches[0].parent)
    raise FileNotFoundError(
        "Could not find priorities.csv in the provided location. "
        "Ensure the bundle contains priorities.csv, boundaries.csv, events.csv, and neighbourhoods.csv."
    )


def _write_kaggle_json(username: str, key: str) -> None:
    """Persist Kaggle credentials in the expected location."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    cred_path = kaggle_dir / "kaggle.json"
    creds = {"username": username, "key": key}
    try:
        existing = json.loads(cred_path.read_text())
    except Exception:
        existing = {}
    if existing != creds:
        cred_path.write_text(json.dumps(creds))
        try:
            os.chmod(cred_path, 0o600)
        except Exception:
            pass


def _get_kaggle_credentials() -> tuple[str | None, str | None]:
    secrets = {}
    try:
        secrets = st.secrets.get("kaggle", {})
    except Exception:
        secrets = {}
    username = (secrets or {}).get("username") or os.environ.get("KAGGLE_USERNAME")
    key = (secrets or {}).get("key") or os.environ.get("KAGGLE_KEY")
    return username, key


def _ensure_kaggle_auth() -> tuple[str, str]:
    username, key = _get_kaggle_credentials()
    if not username or not key:
        raise RuntimeError(
            "Missing Kaggle credentials. Add them via Streamlit secrets (kaggle.username / kaggle.key) "
            "or environment variables KAGGLE_USERNAME / KAGGLE_KEY."
        )
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    _write_kaggle_json(username, key)
    return username, key


@st.cache_resource(show_spinner=True)
def download_kaggle_dataset(slug: str) -> str:
    """Download a Kaggle dataset and return the extraction directory."""
    if not slug:
        raise ValueError("Please provide a Kaggle dataset slug.")
    target = Path(tempfile.mkdtemp(prefix="police_topics_kaggle_"))
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_download_files(slug, path=str(target), unzip=True, quiet=True)
    except Exception as exc:  # pragma: no cover - delegate to UI
        raise RuntimeError(f"Failed to download dataset '{slug}': {exc}") from exc
    return str(target)


@st.cache_data(show_spinner=True)
def load_data(folder: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four CSVs from the resolved folder."""
    directory = Path(folder)

    def _read_csv(name: str) -> pd.DataFrame:
        path = directory / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    neighbourhoods = _read_csv("neighbourhoods.csv")
    boundaries = _read_csv("boundaries.csv")
    events = _read_csv("events.csv")
    priorities = _read_csv("priorities.csv")
    return neighbourhoods, boundaries, events, priorities


# ---------------------------------------------------------------------------
# Sidebar setup
# ---------------------------------------------------------------------------
st.sidebar.title("Controls")

has_creds = all(_get_kaggle_credentials())
data_source = st.sidebar.radio(
    "Data source",
    options=["Download from Kaggle", "Use bundled sample dataset"],
    index=0 if has_creds else 1,
)

dataset_slug = st.sidebar.text_input(
    "Kaggle dataset slug",
    value=os.environ.get("KAGGLE_DATASET", "crimsoneer/uk-police-neighbourhoods-priorities-and-events"),
    help="Format owner/dataset. Requires Kaggle API credentials in Streamlit secrets.",
)

with st.sidebar.expander("Model options"):
    min_topic_size = st.slider("Minimum topic size", 10, 200, 30, step=5)
    nr_topics_mode = st.selectbox("Topic reduction", ["auto", "none"], index=0)

with st.sidebar.expander("Filters"):
    date_from = st.date_input("From date", value=None)
    date_to = st.date_input("To date", value=None)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Set kaggle.username and kaggle.key in Streamlit secrets to download automatically. "
    "Switch to the bundled sample if you just want a quick demo."
)


# ---------------------------------------------------------------------------
# Resolve data directory
# ---------------------------------------------------------------------------
def resolve_data_directory() -> Path:
    if data_source == "Use bundled sample dataset":
        return DEFAULT_DATA_DIR
    try:
        _ensure_kaggle_auth()
    except Exception as exc:
        st.warning(f"{exc}\nFalling back to bundled sample dataset.", icon="⚠️")
        return DEFAULT_DATA_DIR
    slug = dataset_slug.strip()
    if not slug:
        st.warning("Empty Kaggle slug, using bundled sample dataset instead.", icon="⚠️")
        return DEFAULT_DATA_DIR
    try:
        return Path(download_kaggle_dataset(slug))
    except Exception as exc:
        st.error(exc)
        st.stop()


try:
    base_dir = resolve_data_directory()
    data_dir = Path(_locate_dataset_folder(str(base_dir)))
except Exception as exc:  # Catch both ValueError and FileNotFoundError
    st.error(str(exc))
    st.stop()

st.success(f"Using data from: {data_dir}")


# ---------------------------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------------------------
df_neigh, df_bounds, df_events, df_prior = load_data(str(data_dir))
if df_prior.empty:
    st.error("No priorities found in the dataset.")
    st.stop()

def parse_date(value: object) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        return pd.to_datetime(text, utc=True, errors="coerce")
    except Exception:
        return pd.NaT

# Ensure required columns exist
for name in ["issue", "action", "issue_date", "action_date", "scrape_date", "force_id", "neighbourhood_id"]:
    if name not in df_prior.columns:
        df_prior[name] = np.nan

df_prior["issue_date_dt"] = df_prior["issue_date"].apply(parse_date)
df_prior["action_date_dt"] = df_prior["action_date"].apply(parse_date)
df_prior["scrape_date_dt"] = df_prior["scrape_date"].apply(parse_date)

def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    return str(text).replace("\n", " ").replace("\r", " ").strip()

df_prior["text"] = (df_prior["issue"].apply(clean_text) + " " + df_prior["action"].apply(clean_text)).str.strip()

if date_from:
    df_prior = df_prior[
        df_prior[["issue_date_dt", "action_date_dt", "scrape_date_dt"]].min(axis=1)
        >= pd.Timestamp(date_from, tz="UTC")
    ]
if date_to:
    df_prior = df_prior[
        df_prior[["issue_date_dt", "action_date_dt", "scrape_date_dt"]].max(axis=1)
        <= pd.Timestamp(date_to, tz="UTC")
    ]

texts_df = df_prior[df_prior["text"].str.len() > 20].copy()
if texts_df.empty:
    st.error("No suitable texts after applying the filters.")
    st.stop()

centroids = pd.DataFrame(columns=["force_id", "neighbourhood_id", "lat", "lon"])
if not df_bounds.empty and {"latitude", "longitude"}.issubset(df_bounds.columns):
    df_bounds["latitude"] = pd.to_numeric(df_bounds["latitude"], errors="coerce")
    df_bounds["longitude"] = pd.to_numeric(df_bounds["longitude"], errors="coerce")
    centroids = (
        df_bounds.dropna(subset=["latitude", "longitude"])
        .groupby(["force_id", "neighbourhood_id"], as_index=False)[["latitude", "longitude"]]
        .mean()
        .rename(columns={"latitude": "lat", "longitude": "lon"})
    )


# ---------------------------------------------------------------------------
# Topic modelling
# ---------------------------------------------------------------------------
def _build_embeddings(documents: list[str]) -> np.ndarray:
    """Create lightweight TF-IDF based embeddings suitable for BERTopic."""
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(documents)
    dense = matrix.astype(np.float32).toarray()
    if dense.shape[0] < 2:
        return dense
    max_components = min(100, dense.shape[1], dense.shape[0] - 1)
    if max_components >= 2:
        reducer = TruncatedSVD(n_components=max_components, random_state=42)
        reduced = reducer.fit_transform(dense)
        if reduced.shape[1] >= 2:
            return reduced.astype(np.float32)
    # Ensure at least two dimensions for UMAP/HDBSCAN downstream
    if dense.shape[1] < 2:
        pad_width = 2 - dense.shape[1]
        dense = np.hstack([dense, np.zeros((dense.shape[0], pad_width), dtype=np.float32)])
    return dense


@st.cache_resource(show_spinner=True)
def fit_bertopic(
    documents: tuple[str, ...],
    min_size: int,
    topic_mode: str,
) -> tuple[BERTopic, list[int]]:
    docs_list = list(documents)
    embeddings = _build_embeddings(docs_list)
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer,
        min_topic_size=min_size,
        nr_topics=None if topic_mode == "none" else "auto",
        top_n_words=10,
        calculate_probabilities=False,
        low_memory=True,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs_list, embeddings=embeddings)
    return model, topics


with st.spinner("Fitting topics… (cached after first run)"):
    topic_model, topics = fit_bertopic(
        documents=tuple(texts_df["text"].tolist()),
        min_size=min_topic_size,
        topic_mode=nr_topics_mode,
    )

info = topic_model.get_topic_info()
label_col = "Representation" if "Representation" in info.columns else ("Name" if "Name" in info.columns else None)
if label_col:
    info = info.rename(columns={label_col: "label"})
else:
    info["label"] = info["Topic"].apply(
        lambda tid: " / ".join([word for (word, _score) in (topic_model.get_topic(tid) or [])[:3]]) if tid != -1 else "Outliers"
    )

texts_df = texts_df.reset_index(drop=True)
texts_df["topic"] = topics

df_topics = texts_df[
    ["force_id", "neighbourhood_id", "issue_date_dt", "action_date_dt", "scrape_date_dt", "text", "topic"]
].copy()
df_topics = df_topics.merge(info[["Topic", "label"]], left_on="topic", right_on="Topic", how="left")
df_topics["best_date"] = df_topics[["action_date_dt", "issue_date_dt", "scrape_date_dt"]].bfill(axis=1).iloc[:, 0]
df_topics = df_topics.merge(centroids, on=["force_id", "neighbourhood_id"], how="left")


# ---------------------------------------------------------------------------
# KPI header
# ---------------------------------------------------------------------------
st.title("UK Police Priorities — Topics & Geo Dashboard")
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Total items", f"{len(df_topics):,}")
with col_b:
    topic_count = df_topics["topic"].nunique() - (1 if (df_topics["topic"] == -1).any() else 0)
    st.metric("Topics (excl. outliers)", f"{max(topic_count, 0):,}")
with col_c:
    st.metric("Forces", f"{df_topics['force_id'].nunique():,}")
with col_d:
    last_dt = df_topics["best_date"].max()
    st.metric("Last update", last_dt.strftime("%Y-%m-%d") if pd.notna(last_dt) else "—")


# ---------------------------------------------------------------------------
# Topic selection
# ---------------------------------------------------------------------------
topic_counts = (
    df_topics.groupby(["topic", "label"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
)

if topic_counts.empty:
    st.warning("No topics discovered.")
    st.stop()

selected_label = st.selectbox("Pick a topic to explore", topic_counts["label"].tolist(), index=0)
selected_topic = int(topic_counts[topic_counts["label"] == selected_label]["topic"].iloc[0])

words = topic_model.get_topic(selected_topic) or []
word_summary = ", ".join([w for (w, _) in words[:15]]) if selected_topic != -1 else "(Outliers)"
st.caption(f"Top words: {word_summary}")


# ---------------------------------------------------------------------------
# Map visualisation
# ---------------------------------------------------------------------------
st.subheader("Map: Topic distribution")

map_color_mode = st.radio("Map colour", ["Topic label", "Days since update"], horizontal=True)
map_df = df_topics.copy()
map_df["snippet"] = map_df["text"].fillna("").str.slice(0, 160) + np.where(map_df["text"].fillna("").str.len() > 160, "…", "")

has_geo = map_df["lat"].notna().sum() > 2
if has_geo:
    if map_color_mode == "Topic label":
        fig_map = px.scatter_mapbox(
            map_df.dropna(subset=["lat", "lon"]).sample(frac=1.0, random_state=42),
            lat="lat",
            lon="lon",
            color="label",
            hover_data={"force_id": True, "neighbourhood_id": True, "label": True, "snippet": True},
            zoom=5,
            height=600,
        )
    else:
        tmp = map_df.dropna(subset=["lat", "lon"]).copy()
        tmp["best_date"] = pd.to_datetime(tmp["best_date"], utc=True)
        tmp["days_since"] = (pd.Timestamp.now(tz="UTC") - tmp["best_date"]).dt.days
        fig_map = px.scatter_mapbox(
            tmp.sample(frac=1.0, random_state=42),
            lat="lat",
            lon="lon",
            color="days_since",
            color_continuous_scale="Viridis_r",
            hover_data={"force_id": True, "neighbourhood_id": True, "days_since": True, "label": True, "snippet": True},
            zoom=5,
            height=600,
        )
    fig_map.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No neighbourhood coordinates available — showing a force-level treemap instead.")
    fallback = map_df.groupby(["force_id", "label"], dropna=False).size().reset_index(name="count")
    st.plotly_chart(px.treemap(fallback, path=["force_id", "label"], values="count", height=600), use_container_width=True)


# ---------------------------------------------------------------------------
# Timeline views
# ---------------------------------------------------------------------------
st.subheader("Timeline: Topic prevalence over time")

timeline_df = df_topics.dropna(subset=["best_date"]).copy()
timeline_df["month"] = pd.to_datetime(timeline_df["best_date"]).dt.to_period("M").dt.to_timestamp()

monthly = timeline_df.groupby(["month", "label"], as_index=False).size().rename(columns={"size": "count"})
fig_area = px.area(monthly.sort_values("month"), x="month", y="count", color="label", height=450, title="Monthly topic counts")
fig_area.update_layout(xaxis_title="Month", yaxis_title="Count")
st.plotly_chart(fig_area, use_container_width=True)

st.subheader("Per-force intensity (top topics)")
top_labels = topic_counts.head(10)["label"].tolist()
heat = (
    timeline_df[timeline_df["label"].isin(top_labels)]
    .groupby(["force_id", "label", "month"], as_index=False)
    .size()
)
fig_heat = px.density_heatmap(
    heat,
    x="month",
    y="force_id",
    z="size",
    facet_row="label",
    nbinsx=max(6, heat["month"].nunique()) if not heat.empty else 6,
    color_continuous_scale="Blues",
    height=900,
)
fig_heat.update_layout(xaxis_title="Month", yaxis_title="Force")
st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------------
# Drill-down table
# ---------------------------------------------------------------------------
st.subheader("Examples in selected topic")
filtered_rows = df_topics[df_topics["topic"] == selected_topic].copy()
filtered_rows["Best date"] = pd.to_datetime(filtered_rows["best_date"]).dt.strftime("%Y-%m-%d")
st.dataframe(
    filtered_rows[["force_id", "neighbourhood_id", "Best date", "text"]].rename(
        columns={"force_id": "Force", "neighbourhood_id": "Neighbourhood", "text": "Text"}
    ),
    use_container_width=True,
    hide_index=True,
)
