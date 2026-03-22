"""
SautiNet — Streamlit Dashboard
================================
Decentralized Sentiment Analysis for Kenyan Social Media
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
WS_URL   = "ws://localhost:8000/ws/feed"

st.set_page_config(
    page_title="SautiNet",
    page_icon="🇰🇪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared WebSocket queue ──
if "ws_queue" not in st.session_state:
    st.session_state.ws_queue = queue.Queue(maxsize=50)
if "ws_thread_started" not in st.session_state:
    st.session_state.ws_thread_started = False
if "feed_items" not in st.session_state:
    st.session_state.feed_items = []


# ══════════════════════════════════════════════════════
# API Helpers
# ══════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════
# WebSocket background thread
# ══════════════════════════════════════════════════════

def _ws_thread(q: queue.Queue):
    def on_message(ws_app, message):
        try:
            data = json.loads(message)
            if not q.full():
                q.put(data)
        except Exception:
            pass

    def on_error(ws_app, error):
        pass

    def on_close(ws_app, *args):
        pass

    while True:
        try:
            ws_app = websocket.WebSocketApp(
                WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws_app.run_forever(ping_interval=20, ping_timeout=10)
        except Exception:
            pass
        time.sleep(5)  # Reconnect delay


def ensure_ws_thread():
    if not st.session_state.ws_thread_started:
        t = threading.Thread(
            target=_ws_thread,
            args=(st.session_state.ws_queue,),
            daemon=True,
        )
        t.start()
        st.session_state.ws_thread_started = True


# ══════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.title("🇰🇪 SautiNet")
    st.caption("Decentralized Sentiment Analysis\nfor Kenyan Social Media")
    st.divider()

    # Node health
    health = api_get("/api/v1/health")
    if health:
        st.success("Backend: Online")
        st.write(f"**Node:** {health.get('node_id', 'N/A')}")
        st.write(f"**Region:** {health.get('region', 'N/A').title()}")
        st.write(f"**Posts processed:** {health.get('posts_processed', 0):,}")
        st.write(f"**Model loaded:** {'✅' if health.get('model_loaded') else '❌'}")
    else:
        st.error("Backend: Offline")
        st.caption(f"Ensure backend is running at\n`{API_BASE}`")

    st.divider()

    # Navigation
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "📝 Analyze Text", "🗺️ Counties", "📡 Live Feed"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("SautiNet · Distributed ML Project")


# ══════════════════════════════════════════════════════
# Page: Dashboard
# ══════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.title("📊 Dashboard")

    stats = api_get("/api/v1/stats")
    trending = api_get("/api/v1/trending", {"limit": 10})

    # ── KPI row ──
    if stats:
        pipeline = stats.get("pipeline", {})
        worker   = stats.get("worker", {})

        total     = worker.get("processed_count", 0)
        dist      = worker.get("sentiment_distribution", {})
        pos_count = dist.get("positive", 0)
        neg_count = dist.get("negative", 0)
        neu_count = dist.get("neutral", 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed", f"{total:,}")
        c2.metric("Positive 😊", f"{pos_count:,}")
        c3.metric("Neutral 😐", f"{neu_count:,}")
        c4.metric("Negative 😞", f"{neg_count:,}")

        st.divider()

        # ── Sentiment donut + language bar ──
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Sentiment Distribution")
            if total > 0:
                fig = go.Figure(go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[pos_count, neu_count, neg_count],
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
                st.info("No data yet — start the backend and ingest some posts.")

        with col_right:
            st.subheader("Language Breakdown")
            lang_dist = worker.get("language_distribution", {})
            if lang_dist:
                df_lang = pd.DataFrame([
                    {"Language": LANG_LABELS.get(k, k), "Count": v}
                    for k, v in lang_dist.items()
                ])
                fig2 = px.bar(
                    df_lang, x="Language", y="Count",
                    color="Language",
                    color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b"],
                )
                fig2.update_layout(
                    showlegend=False,
                    margin=dict(t=10, b=10, l=10, r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=300,
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No language data yet.")
    else:
        st.warning("Could not load stats. Is the backend running?")

    st.divider()

    # ── Trending topics ──
    st.subheader("🔥 Trending Topics")
    if trending and trending.get("topics"):
        topics = trending["topics"][:10]
        df_topics = pd.DataFrame(topics)

        # Normalise column names — backend may return different shapes
        if "topic" in df_topics.columns:
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


# ══════════════════════════════════════════════════════
# Page: Analyze Text
# ══════════════════════════════════════════════════════

elif page == "📝 Analyze Text":
    st.title("📝 Analyze Text")
    st.caption("Enter text in English, Swahili, or Sheng to analyze sentiment.")

    # ── Input ──
    sample_texts = [
        "Manze hii serikali iko rada wasee mambo ni poa",
        "The government needs to address the rising cost of living urgently",
        "Serikali imeshindwa kusimamia uchumi wa nchi yetu",
        "Fuel prices are too high, wananchi wanaumia sana",
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
        with st.spinner("Analyzing..."):
            result = api_post("/api/v1/analyze", {"text": text_input.strip()})

        if "error" in result:
            st.error(f"Analysis failed: {result['error']}")
        else:
            st.divider()

            # ── Top metrics ──
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
            m4.metric("Topic", topic.title(), "🏛️ Political" if is_pol else "")

            # ── Sentiment gauge ──
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

            # ── Entities & Sheng ──
            col_ent, col_sheng = st.columns(2)

            with col_ent:
                entities = result.get("entities", {})
                flat_entities = []
                for etype, items in entities.items():
                    if items:
                        for item in items:
                            flat_entities.append({"Type": etype.title(), "Entity": item})
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
                    st.write(" · ".join([f"`{w}`" for w in sheng_words]))
                    if code_switch:
                        st.caption("⚡ Code-switching detected")

            st.caption(f"Processed in {proc_ms:.1f} ms")

    elif analyze_btn:
        st.warning("Please enter some text first.")

    # ── Batch analysis ──
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
                        "Topic":     r.get("topics", {}).get("primary_topic", "?"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No texts to analyze.")


# ══════════════════════════════════════════════════════
# Page: Counties
# ══════════════════════════════════════════════════════

elif page == "🗺️ Counties":
    st.title("🗺️ County Sentiment")

    data = api_get("/api/v1/counties")

    if data and data.get("counties"):
        counties = data["counties"]

        df = pd.DataFrame([{
            "County":    c.get("name", "Unknown"),
            "Sentiment": round(c.get("avg_sentiment", 0), 3),
            "Posts":     c.get("post_count", 0),
            "Label":     c.get("dominant_sentiment", "neutral"),
        } for c in counties])

        # ── Search filter ──
        search = st.text_input("🔍 Search counties", placeholder="e.g. Nairobi")
        if search:
            df = df[df["County"].str.contains(search, case=False)]

        # ── Sentiment bar chart ──
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
            height=max(400, len(df_sorted) * 22),
            margin=dict(t=10, b=10, l=10, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            xaxis=dict(range=[-1, 1], zeroline=True, zerolinecolor="#475569"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Table ──
        st.subheader("County Data Table")
        st.dataframe(
            df.sort_values("Posts", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("No county data available yet. The backend needs to process some posts first.")


# ══════════════════════════════════════════════════════
# Page: Live Feed
# ══════════════════════════════════════════════════════

elif page == "📡 Live Feed":
    st.title("📡 Live Sentiment Feed")
    st.caption("Real-time posts as they are analyzed by the pipeline.")

    ensure_ws_thread()

    # Drain queue into session state
    while not st.session_state.ws_queue.empty():
        try:
            item = st.session_state.ws_queue.get_nowait()
            st.session_state.feed_items.insert(0, item)
        except queue.Empty:
            break

    # Keep only last 30
    st.session_state.feed_items = st.session_state.feed_items[:30]

    col_ctrl, col_refresh = st.columns([4, 1])
    with col_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    if not st.session_state.feed_items:
        st.info("Waiting for live data... Make sure the backend is running and ingesting posts.")
    else:
        for item in st.session_state.feed_items:
            sentiment = item.get("sentiment", {}).get("label", "neutral")
            text      = item.get("text", "No text")
            lang_code = item.get("language", {}).get("detected_language", "?")
            topic     = item.get("topics", {}).get("primary_topic", "general")
            score     = item.get("sentiment", {}).get("score", 0)

            border_color = SENTIMENT_COLORS.get(sentiment, "#94a3b8")

            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.write(text[:200] + ("..." if len(text) > 200 else ""))
                    st.caption(
                        f"Lang: **{LANG_LABELS.get(lang_code, lang_code.upper())}** · "
                        f"Topic: **{topic.title()}** · "
                        f"Score: **{score:.2f}**"
                    )
                with c2:
                    st.write(sentiment_badge(sentiment))

    # Auto-refresh every 3 seconds
    time.sleep(3)
    st.rerun()
