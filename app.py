"""
ATLAS — Streamlit Chat Interface

A clean chat UI for interacting with ATLAS in the browser.
Deployed on Streamlit Cloud so anyone can try it via a link.

Run locally:
    streamlit run app.py
"""

import streamlit as st
from agent.core import Agent


# --- Page Configuration ---
st.set_page_config(
    page_title="ATLAS — AI Agent",
    page_icon="🌍",
    layout="centered",
)

# --- Header ---
st.title("🌍 ATLAS")
st.caption('"I carry the weight so you don\'t have to."')
st.caption("A multi-tool AI agent by Venkata Krishna Raj Abhishek Gade")

# --- Initialize Agent (cached so it persists across rerenders) ---
# st.session_state keeps the agent alive across Streamlit rerenders.
# Without this, a new Agent is created on every interaction and
# conversation history is lost.
if "agent" not in st.session_state:
    st.session_state.agent = Agent()
    st.session_state.messages = []

agent = st.session_state.agent

# --- Sidebar ---
with st.sidebar:
    st.header("About ATLAS")
    st.write(
        "ATLAS is a multi-tool AI agent that chains tools together "
        "to answer complex questions."
    )

    st.subheader("Available Tools")
    for tool_name in agent.registry.list_tools():
        tool = agent.registry.get(tool_name)
        st.write(f"🔧 **{tool_name}**: {tool.description[:80]}...")

    st.subheader("Safety")
    blocked_count = len(agent.guardrails.list_blocked_topics())
    st.write(f"🛡️ {blocked_count} blocked topic categories active")

    st.divider()

    if st.button("🔄 Clear Conversation"):
        agent.reset()
        st.session_state.messages = []
        st.rerun()

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask ATLAS anything..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.chat(prompt)
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
