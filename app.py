"""
ATLAS — Streamlit Chat Interface (Phase 6)

Changes:
- Thinking expander: reasoning is hidden by default, viewable via toggle
- Empty input is blocked at the UI level
- Clean message display with no leaked <think> tags
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

# --- Initialize Agent ---
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
        # Show thinking expander for messages that had reasoning
        if msg["role"] == "assistant" and msg.get("thinking"):
            with st.expander("💭 See ATLAS's reasoning"):
                st.markdown(f"```\n{msg['thinking']}\n```")

# --- Chat Input ---
if prompt := st.chat_input("Ask ATLAS anything..."):
    # Block empty/whitespace input at the UI level
    cleaned_prompt = prompt.strip()
    if not cleaned_prompt:
        st.toast("Please type a message first!", icon="⚠️")
    else:
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": cleaned_prompt,
        })
        with st.chat_message("user"):
            st.write(cleaned_prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.chat(cleaned_prompt)

            # Display the answer
            st.write(response)

            # Show thinking expander if reasoning was captured
            thinking = agent.last_thinking
            if thinking:
                with st.expander("💭 See ATLAS's reasoning"):
                    st.markdown(f"```\n{thinking}\n```")

        # Save to message history (including thinking for re-display)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "thinking": agent.last_thinking,
        })
