"""
dashboard.py
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.metrics import confusion_matrix

from classical_ml import load_classifier
from preprocessing import load_and_clean_tickets, split_dataset

API_URL = "http://localhost:8000"
LOGO_PATH = str(Path(__file__).parent.parent / "assets" / "NexusSupport-AI_Logo.png")


def get_image_data_url(path: str) -> str:
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


st.set_page_config(page_title="NexusSupport AI Support Studio", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: rgba(10, 24, 43, 0.9);
        --panel-2: rgba(17, 35, 61, 0.9);
        --accent: #46d0ff;
        --accent-2: #8b5cf6;
        --text: #f8fbff;
        --muted: #8ba0bb;
    }
    .stApp {
        background: radial-gradient(circle at top left, rgba(70,208,255,0.18), transparent 28%),
                    linear-gradient(135deg, var(--bg), #0f2440 60%, #111827);
        color: var(--text);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(70,208,255,0.16), rgba(139,92,246,0.12));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 22px 60px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        margin-right: 0.45rem;
        margin-top: 0.35rem;
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }
    .chat-bubble-user {
        background: linear-gradient(90deg, rgba(70,208,255,0.25), rgba(70,208,255,0.15));
        border-radius: 16px 16px 4px 16px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border: 1px solid rgba(70,208,255,0.18);
    }
    .chat-bubble-assistant {
        background: rgba(255,255,255,0.06);
        border-radius: 16px 16px 16px 4px;
        padding: 0.8rem 1rem;
        margin: 0.3rem 0;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def call_support_api(question: str) -> tuple[str, list[str]]:
    try:
        response = requests.post(f"{API_URL}/ask", json={"text": question}, timeout=8)
        if response.ok:
            payload = response.json()
            return payload.get("answer", "I couldn't generate a response."), payload.get("sources", [])
        return f"The support service returned an error: {response.text}", []
    except requests.RequestException as exc:
        return f"The support service is currently unavailable: {exc}", []


st.markdown(
    f"""
    <div class="hero-card">
        <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
            <img src="{get_image_data_url(LOGO_PATH)}" width="78" height="78" style="border-radius:16px;" />
            <div>
                <div class="hero-title">NexusSupport AI</div>
                <div class="hero-subtitle">A polished support assistant experience for customer questions, model insights, and live service health.</div>
            </div>
        </div>
        <div style="margin-top:0.7rem;">
            <span class="pill">Conversational UI</span>
            <span class="pill">Professional Design</span>
            <span class="pill">Live API Metrics</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tab1, tab2, tab3 = st.tabs(["Ask NexusSupport AI", "Model Performance", "Live API Metrics"])

with tab1:
    col_left, col_right = st.columns([1.4, 0.8])
    with col_left:
        st.markdown("### Ask a support question")
        st.caption("Try questions like: 'How long do refunds take?' or 'Can I merge two accounts?'")
        question = st.text_input(
            "Support question",
            placeholder="Type your question here...",
            label_visibility="collapsed",
        )
        if st.button("Get answer", use_container_width=True):
            if question.strip():
                with st.spinner("Searching the knowledge base..."):
                    answer, sources = call_support_api(question)
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("assistant", answer))
                st.session_state.chat_history.append(("sources", sources))
            else:
                st.warning("Please enter a question first.")

    with col_right:
        st.markdown("### Quick hints")
        st.info("The assistant replies using the latest support knowledge base and the live API.")
        st.markdown("- Billing and refunds")
        st.markdown("- Account changes")
        st.markdown("- Technical troubleshooting")

    st.markdown("---")
    if st.session_state.chat_history:
        for i, item in enumerate(st.session_state.chat_history):
            if item[0] == "user":
                with st.chat_message("user"):
                    st.markdown(f"<div class='chat-bubble-user'>{item[1]}</div>", unsafe_allow_html=True)
            elif item[0] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(f"<div class='chat-bubble-assistant'>{item[1]}</div>", unsafe_allow_html=True)
            else:
                sources = item[1]
                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.write(f"• {source}")
    else:
        st.markdown("Start by typing a question to see the assistant reply in a polished chat-style panel.")

with tab2:
    st.subheader("Classifier Evaluation")
    try:
        df = load_and_clean_tickets("data/support_tickets.csv")
        _, _, test = split_dataset(df)
        pipeline = load_classifier("models/ticket_classifier.joblib")

        preds = pipeline.predict(test["text_clean"])
        labels = sorted(df["category"].unique())
        cm = confusion_matrix(test["category"], preds, labels=labels)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(
                cm,
                x=labels,
                y=labels,
                text_auto=True,
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                title="Confusion Matrix (Test Set)",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            accuracy = (preds == test["category"]).mean()
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("Test Accuracy", f"{accuracy:.1%}")
            counts = df["category"].value_counts().reset_index()
            counts.columns = ["category", "count"]
            fig2 = px.bar(counts, x="category", y="count", title="Class Distribution in Dataset")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.warning(
            "Train the model first: run `python data/generate_sample_data.py` "
            "then `python src/classical_ml.py` from the project root."
        )

with tab3:
    st.subheader("Live Request Metrics")
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=3)
        requests_data = resp.json().get("requests", [])
        if not requests_data:
            st.info("No requests logged yet — call the API a few times, then refresh.")
        else:
            log_df = pd.DataFrame(requests_data)
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Requests", len(log_df))
            col2.metric("Avg Latency (ms)", f"{log_df['latency_ms'].mean():.0f}")
            col3.metric("Endpoints Used", log_df["endpoint"].nunique())

            fig3 = px.histogram(log_df, x="endpoint", title="Requests by Endpoint")
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.box(log_df, x="endpoint", y="latency_ms", title="Latency by Endpoint")
            st.plotly_chart(fig4, use_container_width=True)

    except requests.exceptions.ConnectionError:
        st.error(
            f"Couldn't reach the API at {API_URL}. Start it with:\n\n"
            "`uvicorn src.app:app --reload --port 8000`"
        )
