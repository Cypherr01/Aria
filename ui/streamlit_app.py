"""
ui.streamlit_app
================
Main Streamlit application for the ARIA UI.

Run with: `streamlit run ui/streamlit_app.py`
"""
from __future__ import annotations

import os
import uuid
import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

# Load the project .env so ARIA_API_URL / ARIA_API_KEY are available via os.getenv().
# We intentionally avoid st.secrets: accessing it when no secrets.toml exists causes
# Streamlit to render a permanent error banner in the UI regardless of any try/except.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[1] / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed — fall back to already-set env vars

from utils.api_client import ARIAClient
from config.config import get_config

# Must be the first Streamlit command
st.set_page_config(
    page_title="ARIA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.getenv("ARIA_API_URL", "http://localhost:8000")
API_KEY = os.getenv("ARIA_API_KEY", "")

client = ARIAClient(API_URL, API_KEY)

# ── Session State Initialization ──────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:12]}"
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None
if "primary_model" not in st.session_state:
    _default_chain = get_config().model_tiers.reasoning_chain
    st.session_state.primary_model = _default_chain[0] if _default_chain else None


def reset_session():
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:12]}"
    st.session_state.messages = []
    st.session_state.pending_followup = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ARIA Settings")

    # Section 1: Session
    st.subheader("Session")
    st.code(st.session_state.session_id)
    if st.button("New Session", use_container_width=True):
        reset_session()
        st.rerun()

    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.debug_mode = st.toggle("Debug Mode", value=st.session_state.debug_mode)
    
    _primary_chain = get_config().model_tiers.reasoning_chain
    _cur = st.session_state.primary_model
    _idx = _primary_chain.index(_cur) if _cur in _primary_chain else 0
    st.session_state.primary_model = st.selectbox(
        "Primary Model",
        options=_primary_chain,
        index=_idx,
        help="Applies to synthesis, planning, and reflection only. Background tasks always use the fast chain.",
    )
    _fast_chain = get_config().model_tiers.speed_chain
    st.caption(f"⚡ Fast chain (fixed): {' → '.join(_fast_chain[:2])} …")
    
    st.divider()

    # Section 2: Model Health
    st.subheader("Model Health")
    status_data = client.get_model_status()
    models = status_data.get("models", [])
    
    if models:
        for m in models:
            provider = m.get("provider", "unknown")
            model_name = m.get("model", "unknown")
            healthy = m.get("healthy", False)
            usage = m.get("token_usage", {})
            
            icon = "🟢" if healthy else "🔴"
            st.markdown(f"**{icon} {provider}** | {model_name}")
            
            # Show progress bar if daily_limit is available
            limit = usage.get("daily_limit")
            used = usage.get("used_today", 0)
            if limit and limit > 0:
                fraction = min(used / limit, 1.0)
                st.progress(fraction)
                st.caption(f"Tokens: {used:,} / {limit:,}")
            else:
                st.caption(f"Tokens used: {used:,}")
    else:
        st.caption("Unable to fetch model status.")
        
    st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

    st.divider()

    # Section 3: Knowledge Base
    st.subheader("Knowledge Base")
    uploaded_file = st.file_uploader(
        "Upload Any File",
        type=None,  # All file types accepted — routed by FileRouter
        help="Supported: PDF, DOCX, XLSX, CSV, PPTX, code files, Jupyter notebooks, ZIP archives, and more."
    )
    if uploaded_file and st.button("Ingest Document", use_container_width=True):
        with st.spinner(f"Parsing '{uploaded_file.name}'..."):
            file_bytes = uploaded_file.getvalue()
            res = client.upload_document(file_bytes, uploaded_file.name, st.session_state.user_id)
            if "error" in res:
                st.error(f"Upload failed: {res['error']}")
            else:
                chunk_count = res.get('chunk_count', 0)
                ingestion_ms = res.get('ingestion_time_ms', 0)
                fmt = res.get('format', '').upper()
                st.success(f"✅ Ingested **{chunk_count} chunks** from {fmt} file in {ingestion_ms} ms")

                # Chunk breakdown
                chunk_types = res.get('chunk_types', {})
                if chunk_types:
                    with st.expander("📊 Chunk Breakdown"):
                        for ctype, count in sorted(chunk_types.items(), key=lambda x: -x[1]):
                            st.markdown(f"- **{ctype}**: {count} chunk{'s' if count != 1 else ''}")

                # File summary
                summary = res.get('summary', '')
                if summary:
                    with st.expander("📝 File Summary"):
                        st.markdown(summary)

    with st.expander("Manage Documents"):
        docs = client.get_documents(st.session_state.user_id)
        if docs:
            for d in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.caption(d["filename"])
                with col2:
                    if st.button("🗑️", key=f"del_doc_{d['doc_id']}", help="Delete"):
                        client.delete_document(d["doc_id"], st.session_state.user_id)
                        st.rerun()
        else:
            st.caption("No documents ingested.")


# ── Main Area ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🧠 Memory", "📊 Analytics"])

with tab1:
    st.title("Chat with ARIA")
    st.caption("Agentic Reasoning and Intelligent Assistant")
    
    # Render chat history
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        with st.chat_message(role):
            st.markdown(content)
            
            if role == "assistant":
                # Debug reasoning trace
                plan_trace = msg.get("plan_trace")
                if plan_trace and st.session_state.debug_mode:
                    with st.expander("🧠 Reasoning Trace"):
                        for step in plan_trace:
                            icon = "✅" if step["status"] == "success" else "⏳"
                            st.write(f"{step['step']}. {icon} {step['action']}")
                
                # Confidence indicator / fast-response badge
                conf = msg.get("confidence_indicator")
                model = msg.get("model_used")
                latency = msg.get("total_latency_ms")
                skipped = msg.get("reflection_skipped", False)
                if model:
                    if skipped:
                        st.caption(f"⚡ Fast response | Model: {model} | {latency}ms")
                    elif conf:
                        conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
                        st.caption(f"{conf_color} Confidence: {conf} | Model: {model} | {latency}ms")
                
                # Citations
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"📚 Sources ({len(citations)})"):
                        for c in citations:
                            st.write(f"- {c.get('text')} *(Pos: {c.get('position', 0)})*")
                
                # Follow up suggestions
                suggestions = msg.get("follow_up_suggestions", [])
                if suggestions and msg == st.session_state.messages[-1]:
                    st.write("**Suggested Follow-ups:**")
                    cols = st.columns(len(suggestions))
                    for i, suggestion in enumerate(suggestions):
                        if cols[i].button(suggestion, key=f"sugg_{i}"):
                            st.session_state.pending_followup = suggestion
                            st.rerun()

    # Chat Input
    prompt = st.chat_input("Ask ARIA anything...")
    
    # Use pending_followup if button was clicked
    if st.session_state.pending_followup:
        prompt = st.session_state.pending_followup
        st.session_state.pending_followup = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("🧠 ARIA is thinking..."):
                result = client.chat(
                    message=prompt,
                    session_id=st.session_state.session_id,
                    user_id=st.session_state.user_id,
                    debug_mode=st.session_state.debug_mode,
                    primary_model=st.session_state.primary_model
                )
                
            if "error" in result:
                st.error(result["error"])
                st.session_state.messages.append({"role": "assistant", "content": result["error"]})
            else:
                resp_text = result.get("response", "")
                st.markdown(resp_text)
                
                # Append to messages with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resp_text,
                    "plan_trace": result.get("plan_trace"),
                    "confidence_indicator": result.get("confidence_indicator"),
                    "reflection_skipped": result.get("reflection_skipped", False),
                    "model_used": result.get("model_used"),
                    "total_latency_ms": result.get("total_latency_ms"),
                    "citations": result.get("citations"),
                    "follow_up_suggestions": result.get("follow_up_suggestions")
                })
                st.rerun()


with tab2:
    st.title("Your Long-Term Memory")
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Refresh Memories"):
            pass # Rerun implicitly happens
    with col_b:
        min_importance = st.slider("Min Importance Filter", 0.0, 1.0, 0.0, 0.1)
        
    memories = client.get_memories(st.session_state.user_id, min_importance)
    
    if not memories:
        st.info("No memories stored yet. Start chatting to build memory.")
    else:
        high_importance = sum(1 for m in memories if m.get("importance_score", 0) >= 0.7)
        st.write(f"**ARIA has {len(memories)} memories about you, {high_importance} high importance.**")
        
        for m in memories:
            with st.container(border=True):
                st.write(m["content"])
                
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.caption(f"Category: {m.get('category', 'N/A')}")
                with c2: st.caption(f"Score: {m.get('importance_score', 0):.2f}")
                with c3: st.caption(f"Accessed: {m.get('access_count', 0)}")
                with c4:
                    if st.button("Delete", key=f"del_mem_{m['memory_id']}"):
                        client.delete_memory(m["memory_id"], st.session_state.user_id)
                        st.rerun()


with tab3:
    st.title("Usage Analytics")
    period = st.radio("Period", ["7 days", "30 days"], horizontal=True)
    days = 7 if period == "7 days" else 30
    
    analytics = client.get_analytics(days)
    
    if not analytics:
        st.info("Analytics API is unreachable.")
    else:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total API Latency (ms)", f"{analytics.get('avg_response_latency_ms', 0):.0f}")
        
        failure_rates = analytics.get("failure_rates", [])
        if failure_rates:
            col2.metric("Component Failures", sum(f.get("errors", 0) for f in failure_rates))
        else:
            col2.metric("Component Failures", 0)
            
        token_usage = analytics.get("token_usage_by_day", [])
        total_tokens = sum(t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in token_usage)
        col3.metric("Tokens Consumed", f"{total_tokens:,}")
        
        intents = analytics.get("intent_distribution", [])
        col4.metric("Unique Intents", len(intents))
        
        st.divider()
        
        # Tool usage table
        st.subheader("Tool Usage")
        tools = analytics.get("most_used_tools", [])
        if tools:
            df = pd.DataFrame(tools)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No tool usage data.")
            
        # Token usage chart
        st.subheader("Token Usage Over Time")
        if token_usage:
            df_tokens = pd.DataFrame(token_usage)
            if "date" in df_tokens.columns:
                df_tokens.set_index("date", inplace=True)
                st.bar_chart(df_tokens[["input_tokens", "output_tokens"]])
            else:
                st.caption("Invalid token format.")
        else:
            st.caption("No token data.")
