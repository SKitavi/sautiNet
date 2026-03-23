"""
SautiNet — Streamlit Dashboard (Fixed)
=========================================
Decentralized Sentiment Analysis for Kenyan Social Media

Fixes applied:
- Dashboard & Counties auto-refresh every N seconds
- Schema alignment (backend field names → frontend expectations)
- WebSocket connection state tracking & reconnect indicator
- Live Feed drain stability
- Ingestion status panel
"""

import time
import threading
import queue
import json

import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import websocket

# ── Config ──
API_BASE = "http://localhost:8000"
WS_URL   = "ws://localhost:8000/ws/all"  # FIX: subscribe to ALL channels

DASHBOARD_REFRESH_SECS = 5
LIVE_FEED_REFRESH_SECS = 2

st.set_page_config(
    page_title="SautiNet",
    page_icon="\U0001f1f0\U0001f1ea",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ──
if "ws_queue" not in st.session_state:
    st.session_state.ws_queue = queue.Queue(maxsize=200)
if "ws_thread_started" not in st.session_state:
    st.session_state.ws_thread_started = False
if "ws_connected" not in st.session_state:
    st.session_state.ws_connected = False
if "feed_items" not in st.session_state:
    st.session_state.feed_items = []
if "county_cache" not in st.session_state:
    st.session_state.county_cache = []
if "stats_cache" not in st.session_state:
    st.session_state.stats_cache = {}
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0


# ==============================================================
# API Helpers
# ==============================================================

def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(path: str, payload: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ==============================================================
# WebSocket background thread — subscribes to ALL channels
# ==============================================================

def _ws_thread(q: queue.Queue):
    """Background thread that maintains a persistent WebSocket connection."""

    def on_open(ws_app):
        try:
            q.put({"_ws_event": "connected"}, block=False)
        except queue.Full:
            pass

    def on_message(ws_app, message):
        try:
            data = json.loads(message)
            if not q.full():
                q.put(data, block=False)
        except Exception:
            pass

    def on_error(ws_app, error):
        try:
            q.put({"_ws_event": "error", "detail": str(error)}, block=False)
        except queue.Full:
            pass

    def on_close(ws_app, close_status_code, close_msg):
        try:
            q.put({"_ws_event": "disconnected"}, block=False)
        except queue.Full:
            pass

    while True:
        try:
            ws_app = websocket.WebSocketApp(
                WS_URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws_app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception:
            pass
        time.sleep(3)


def ensure_ws_thread():
    if not st.session_state.ws_thread_started:
        t = threading.Thread(
            target=_ws_thread,
            args=(st.session_state.ws_queue,),
            daemon=True,
        )
        t.start()
        st.session_state.ws_thread_started = True


def drain_ws_queue():
    """Drain all WebSocket messages into session state, handling control events."""
    while not st.session_state.ws_queue.empty():
        try:
            item = st.session_state.ws_queue.get_nowait()
        except queue.Empty:
            break

        # Handle internal WS control events
        if "_ws_event" in item:
            event = item["_ws_event"]
            if event == "connected":
                st.session_state.ws_connected = True
            elif event in ("disconnected", "error"):
                st.session_state.ws_connected = False
            continue

        # Route by channel
        channel = item.get("channel", "")
        data = item.get("data", item)

        if channel == "feed":
            st.session_state.feed_items.insert(0, data)
            st.session_state.feed_items = st.session_state.feed_items[:50]

        elif channel == "counties":
            county_name = data.get("county")
            if county_name:
                existing = [c for c in st.session_state.county_cache if c.get("county") != county_name]
                existing.append(data)
                st.session_state.county_cache = existing

        elif channel == "stats":
            st.session_state.stats_cache = data


# ==============================================================
# Helpers
# ==============================================================

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#94a3b8",
}

LANG_LABELS = {"en": "English", "sw": "Swahili", "sh": "Sheng"}


def sentiment_badge(label: str) -> str:
    colors = {"positive": "green", "negative": "red", "neutral": "gray"}
    c = colors.get(label, "gray")
    return f":{c}[**{label.upper()}**]"


def ws_status_indicator():
    if st.session_state.ws_connected:
        st.success("Live feed: Connected")
    else:
        st.warning("Live feed: Reconnecting...")


def score_to_label(score: float) -> str:
    if score > 0.25:
        return "positive"
    elif score < -0.25:
        return "negative"
    return "neutral"


# ==============================================================
# Start WebSocket + drain on every page load
# ==============================================================

ensure_ws_thread()
drain_ws_queue()


# ==============================================================
# Sidebar
# ==============================================================

with st.sidebar:
    st.title("\U0001f1f0\U0001f1ea SautiNet")
    st.caption("Decentralized Sentiment Analysis\nfor Kenyan Social Media")
    st.divider()

    health = api_get("/api/v1/health")
    if health:
        st.success("Backend: Online")
        st.write(f"**Node:** {health.get('node_id', 'N/A')}")
        st.write(f"**Region:** {health.get('region', 'N/A').title()}")
        st.write(f"**Posts processed:** {health.get('posts_processed', 0):,}")
        model_loaded = health.get("model_loaded", False)
        st.write(f"**Model loaded:** {'yes' if model_loaded else 'no (rule-based fallback)'}")
    else:
        st.error("Backend: Offline")
        st.caption(f"Ensure backend is running at\n`{API_BASE}`")

    st.divider()
    ws_status_indicator()
    st.divider()

    ingestion = api_get("/api/v1/ingestion/status")
    if ingestion and ingestion.get("running"):
        connectors = ingestion.get("connectors", {})
        active = [name for name, info in connectors.items() if info.get("task_running")]
        if active:
            st.info(f"Ingesting from: {', '.join(active)}")
        total_ingested = ingestion.get("global", {}).get("total_ingested", 0)
        if total_ingested:
            st.caption(f"Posts ingested: {total_ingested:,}")

    st.divider()

    page = st.radio(
        "Navigate",
        ["Dashboard", "Analyze Text", "Counties", "Live Feed"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("SautiNet - Distributed ML Project")


# ==============================================================
# Page: Dashboard (auto-refreshes)
# ==============================================================

if page == "Dashboard":
    st.title("Dashboard")

    stats = api_get("/api/v1/stats")
    trending = api_get("/api/v1/trending", {"limit": 10})

    if stats:
        pipeline = stats.get("pipeline", {})
        worker   = stats.get("worker", {})
        agg      = worker.get("aggregator_stats", {})

        total     = worker.get("processed_count", 0)
        pos_pct  = agg.get("positive_pct", 0)
        neg_pct  = agg.get("negative_pct", 0)
        neu_pct  = agg.get("neutral_pct", 0)
        window   = agg.get("window_posts", 0)
        rate     = agg.get("processing_rate_per_min", 0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Processed", f"{total:,}")
        c2.metric("Window Posts", f"{window:,}", f"{rate:.1f}/min")
        c3.metric("Positive", f"{pos_pct:.1f}%")
        c4.metric("Neutral", f"{neu_pct:.1f}%")
        c5.metric("Negative", f"{neg_pct:.1f}%")

        st.divider()

        active_model = pipeline.get("active_model", "unknown")
        custom_loaded = pipeline.get("custom_model_loaded", False)
        model_info = f"**Active model:** `{active_model}`"
        if custom_loaded:
            model_info += " + BiLSTM ensemble"
        st.caption(model_info)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Sentiment Distribution")
            if window > 0:
                fig = go.Figure(go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[pos_pct, neu_pct, neg_pct],
                    hole=0.55,
                    marker_colors=[
                        SENTIMENT_COLORS["positive"],
                        SENTIMENT_COLORS["neutral"],
                        SENTIMENT_COLORS["negative"],
                    ],
                    textinfo="label+percent",
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data yet -- start the backend and ingest some posts.")

        with col_right:
            st.subheader("Pipeline Components")
            components = pipeline.get("components", {})
            if components:
                for comp, status in components.items():
                    icon = "ok" if status not in ("inactive", "failed", None) else "x"
                    label = status if isinstance(status, str) else str(status)
                    st.write(f"[{icon}] **{comp.replace('_', ' ').title()}**: `{label}`")
            else:
                st.info("Pipeline info not available.")

        active_counties = agg.get("active_counties", 0)
        active_topics = agg.get("active_topics", 0)
        overall_sent = agg.get("overall_sentiment", 0)

        st.divider()
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Active Counties", active_counties)
        cc2.metric("Active Topics", active_topics)
        cc3.metric("Overall Sentiment", f"{overall_sent:+.3f}")

    else:
        st.warning("Could not load stats. Is the backend running?")

    st.divider()

    st.subheader("Trending Topics")
    if trending and trending.get("topics"):
        topics = trending["topics"][:10]
        df_topics = pd.DataFrame(topics)

        if "topic" in df_topics.columns and "name" not in df_topics.columns:
            df_topics = df_topics.rename(columns={"topic": "name"})
        if "count" not in df_topics.columns and "mentions" in df_topics.columns:
            df_topics = df_topics.rename(columns={"mentions": "count"})

        if "name" in df_topics.columns and "count" in df_topics.columns:
            fig3 = px.bar(
                df_topics.sort_values("count"),
                x="count", y="name",
                orientation="h",
                color="count",
                color_continuous_scale="Blues",
                labels={"count": "Mentions", "name": "Topic"},
            )
            fig3.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
                height=350,
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.json(topics)
    else:
        st.info("No trending topics yet.")

    if stats and stats.get("worker"):
        error_rate = stats["worker"].get("error_rate", 0)
        if error_rate > 5:
            st.warning(f"Worker error rate: {error_rate:.1f}%")

    time.sleep(DASHBOARD_REFRESH_SECS)
    st.rerun()


# ==============================================================
# Page: Analyze Text
# ==============================================================

elif page == "Analyze Text":
    st.title("Analyze Text")
    st.caption("Enter text in English, Swahili, or Sheng to analyze sentiment.")

    sample_texts = [
        "Manze hii serikali iko rada wasee mambo ni poa",
        "The government needs to address the rising cost of living urgently",
        "Serikali imeshindwa kusimamia uchumi wa nchi yetu",
        "Fuel prices are too high, wananchi wanaumia sana",
        "Tech ndio future ya Kenya na youth wako ready",
        "Rushwa ni adui mkubwa wa maendeleo katika nchi yetu",
    ]

    col_input, col_sample = st.columns([3, 1])
    with col_sample:
        st.write("")
        st.write("")
        if st.button("Load sample text"):
            import random
            st.session_state["sample_text"] = random.choice(sample_texts)

    with col_input:
        default_text = st.session_state.get("sample_text", "")
        text_input = st.text_area(
            "Input text",
            value=default_text,
            height=120,
            placeholder="Type or paste text here...",
            label_visibility="collapsed",
        )

    analyze_btn = st.button("Analyze", type="primary", use_container_width=False)

    if analyze_btn and text_input.strip():
        with st.spinner("Running through NLP pipeline..."):
            result = api_post("/api/v1/analyze", {"text": text_input.strip()})

        if "error" in result:
            st.error(f"Analysis failed: {result['error']}")
        else:
            st.divider()

            lang_raw   = result.get("language", {})
            sent_raw   = result.get("sentiment", {})
            topic_raw  = result.get("topics", {})

            lang_code  = lang_raw.get("detected_language", "?")
            lang_label = LANG_LABELS.get(lang_code, lang_code.upper())
            lang_conf  = lang_raw.get("confidence", 0)

            sentiment  = sent_raw.get("label", "neutral")
            sent_score = sent_raw.get("score", 0)
            sent_conf  = sent_raw.get("confidence", 0)
            model_used = sent_raw.get("model_used", "N/A")

            topic      = topic_raw.get("primary_topic", "general")
            is_pol     = topic_raw.get("is_political", False)
            proc_ms    = result.get("processing_time_ms", 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Language", f"{lang_label}", f"{lang_conf*100:.0f}% confidence")
            m2.metric("Sentiment", sentiment.title(), f"score {sent_score:.2f}")
            m3.metric("Confidence", f"{sent_conf*100:.0f}%", model_used)
            m4.metric("Topic", topic.title(), "Political" if is_pol else "")

            st.subheader("Sentiment Score")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sent_score,
                number={"suffix": "", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1},
                    "bar": {"color": SENTIMENT_COLORS.get(sentiment, "#94a3b8")},
                    "steps": [
                        {"range": [-1, -0.1], "color": "#3b0a0a"},
                        {"range": [-0.1, 0.1], "color": "#1e293b"},
                        {"range": [0.1, 1],   "color": "#052e16"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": sent_score,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=250,
                margin=dict(t=20, b=10, l=40, r=40),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            probs = sent_raw.get("probabilities", {})
            if probs:
                st.subheader("Class Probabilities")
                prob_cols = st.columns(3)
                for i, (lbl, val) in enumerate(probs.items()):
                    prob_cols[i].metric(lbl.title(), f"{val*100:.1f}%")

            col_ent, col_sheng = st.columns(2)

            with col_ent:
                entities_raw = result.get("entities", {})
                flat_entities = []
                if isinstance(entities_raw, dict):
                    for key, items in entities_raw.items():
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict):
                                    flat_entities.append({
                                        "Type": item.get("label", key).title(),
                                        "Entity": item.get("text", str(item)),
                                    })
                                elif isinstance(item, str):
                                    flat_entities.append({
                                        "Type": key.replace("_", " ").title(),
                                        "Entity": item,
                                    })

                if flat_entities:
                    st.subheader("Detected Entities")
                    st.dataframe(
                        pd.DataFrame(flat_entities),
                        use_container_width=True,
                        hide_index=True,
                    )

            with col_sheng:
                sheng_words = lang_raw.get("sheng_indicators", [])
                code_switch = lang_raw.get("contains_code_switching", False)
                if sheng_words:
                    st.subheader("Sheng Words Detected")
                    st.write(" - ".join([f"`{w}`" for w in sheng_words]))
                    if code_switch:
                        st.caption("Code-switching detected")

            st.caption(f"Processed in {proc_ms:.1f} ms | Model: `{model_used}`")

    elif analyze_btn:
        st.warning("Please enter some text first.")

    st.divider()
    st.subheader("Batch Analysis")
    st.caption("Paste multiple texts, one per line.")

    batch_input = st.text_area("Batch input", height=150, label_visibility="collapsed",
                               placeholder="Line 1\nLine 2\nLine 3...")
    if st.button("Analyze Batch", type="secondary"):
        texts = [t.strip() for t in batch_input.strip().splitlines() if t.strip()]
        if texts:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                result = api_post("/api/v1/analyze/batch", {"texts": texts})

            if "error" in result:
                st.error(result["error"])
            else:
                rows = []
                for r in result.get("results", []):
                    rows.append({
                        "Text":      r.get("text", "")[:80],
                        "Language":  LANG_LABELS.get(r.get("language", {}).get("detected_language"), "?"),
                        "Sentiment": r.get("sentiment", {}).get("label", "?"),
                        "Score":     round(r.get("sentiment", {}).get("score", 0), 3),
                        "Confidence": f"{r.get('sentiment', {}).get('confidence', 0)*100:.0f}%",
                        "Model":     r.get("sentiment", {}).get("model_used", "?"),
                        "Topic":     r.get("topics", {}).get("primary_topic", "?"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No texts to analyze.")


# ==============================================================
# Page: Counties (auto-refreshes)
# ==============================================================

elif page == "Counties":
    st.title("County Sentiment")

    data = api_get("/api/v1/counties")

    if data and data.get("counties"):
        counties = data["counties"]

        df = pd.DataFrame([{
            "County":    c.get("county", c.get("name", "Unknown")),
            "Sentiment": round(c.get("overall_sentiment", c.get("avg_sentiment", 0)), 3),
            "Posts":     c.get("total_posts", c.get("post_count", 0)),
            "Positive":  c.get("positive_count", 0),
            "Negative":  c.get("negative_count", 0),
            "Neutral":   c.get("neutral_count", 0),
            "Label":     score_to_label(c.get("overall_sentiment", c.get("avg_sentiment", 0))),
            "Trending":  ", ".join(c.get("trending_topics", [])[:3]),
            "Dominant Language": LANG_LABELS.get(
                c.get("dominant_language", "en"), c.get("dominant_language", "en")
            ),
        } for c in counties])

        search = st.text_input("Search counties", placeholder="e.g. Nairobi")
        if search:
            df = df[df["County"].str.contains(search, case=False)]

        kc1, kc2, kc3, kc4 = st.columns(4)
        kc1.metric("Counties Active", len(df))
        kc2.metric("Total Posts", f"{df['Posts'].sum():,}")
        avg_sent = df["Sentiment"].mean() if len(df) > 0 else 0
        kc3.metric("Avg Sentiment", f"{avg_sent:+.3f}")
        most_active = df.loc[df["Posts"].idxmax(), "County"] if len(df) > 0 else "N/A"
        kc4.metric("Most Active", most_active)

        st.divider()

        st.subheader("Sentiment Scores by County")
        df_sorted = df.sort_values("Sentiment", ascending=True)
        colors = [SENTIMENT_COLORS.get(l, "#94a3b8") for l in df_sorted["Label"]]

        fig = go.Figure(go.Bar(
            x=df_sorted["Sentiment"],
            y=df_sorted["County"],
            orientation="h",
            marker_color=colors,
            text=df_sorted["Sentiment"],
            textposition="outside",
        ))
        fig.update_layout(
            height=max(400, len(df_sorted) * 28),
            margin=dict(t=10, b=10, l=10, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(range=[-1, 1], zeroline=True, zerolinecolor="#475569"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("County Data Table")
        st.dataframe(
            df.sort_values("Posts", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No county data available yet. The backend needs to process some posts first.")

    time.sleep(DASHBOARD_REFRESH_SECS)
    st.rerun()


# ==============================================================
# Page: Live Feed
# ==============================================================

elif page == "Live Feed":
    st.title("Live Sentiment Feed")
    st.caption("Real-time posts as they are analyzed by the pipeline.")

    col_status, col_count, col_refresh = st.columns([3, 2, 1])
    with col_status:
        if st.session_state.ws_connected:
            st.success("Connected to live stream")
        else:
            st.warning("Connecting to live stream...")
    with col_count:
        st.caption(f"Feed items: {len(st.session_state.feed_items)}")
    with col_refresh:
        if st.button("Refresh"):
            st.rerun()

    if not st.session_state.feed_items:
        st.info(
            "Waiting for live data...\n\n"
            "Make sure the backend is running and processing posts. "
            "The NLP worker broadcasts results via WebSocket as posts are analyzed."
        )
    else:
        for item in st.session_state.feed_items:
            if "data" in item and isinstance(item["data"], dict):
                item = item["data"]

            sentiment = item.get("sentiment_label", item.get("sentiment", {}).get("label", "neutral") if isinstance(item.get("sentiment"), dict) else "neutral")
            text      = item.get("text", "No text")
            lang_code = item.get("language", "?")
            if isinstance(lang_code, dict):
                lang_code = lang_code.get("detected_language", "?")
            topic     = item.get("topic", "general")
            if isinstance(topic, dict):
                topic = topic.get("primary_topic", "general")
            score     = item.get("sentiment_score", 0)
            if isinstance(score, dict):
                score = 0
            county    = item.get("county", "")
            platform  = item.get("platform", "")
            proc_ms   = item.get("processing_time_ms", 0)

            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.write(text[:250] + ("..." if len(text) > 250 else ""))
                    meta_parts = [
                        f"Lang: **{LANG_LABELS.get(lang_code, str(lang_code).upper())}**",
                        f"Topic: **{topic.title() if isinstance(topic, str) else str(topic)}**",
                        f"Score: **{score:.2f}**",
                    ]
                    if county:
                        meta_parts.append(f"County: **{county}**")
                    if platform:
                        meta_parts.append(f"Platform: {platform}")
                    if proc_ms:
                        meta_parts.append(f"{proc_ms:.0f}ms")
                    st.caption(" | ".join(meta_parts))
                with c2:
                    st.write(sentiment_badge(sentiment))

    time.sleep(LIVE_FEED_REFRESH_SECS)
    st.rerun()
