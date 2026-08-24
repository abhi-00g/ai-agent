"""
Configuration for the AI Agent.

Uses Groq (Qwen 3.6 27B) for LLM inference.
Optionally sends telemetry to the AI Cost Dashboard.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Settings ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "qwen/qwen3.6-27b"

# --- Agent Settings ---
MAX_STEPS = 10

# --- Tool Call Format ---
TOOL_CALL_PREFIX = "TOOL_CALL:"
TOOL_INPUT_PREFIX = "INPUT:"
FINAL_ANSWER_PREFIX = "FINAL_ANSWER:"

# --- Cost Dashboard Telemetry ---
# When both are set, every Groq call is logged to the AI Cost Dashboard.
# When either is missing, telemetry is silently disabled — the agent
# works fine without it. Same graceful fallback pattern as the RAG project.
COST_DASHBOARD_API_KEY = os.getenv("COST_DASHBOARD_API_KEY")
COST_DASHBOARD_ENDPOINT = os.getenv("COST_DASHBOARD_ENDPOINT")

# Qwen 3.6 27B pricing on Groq (as of August 2026)
# Source: https://groq.com/pricing
QWEN_INPUT_PRICE_PER_TOKEN = 0.60 / 1_000_000   # $0.60 per 1M tokens
QWEN_OUTPUT_PRICE_PER_TOKEN = 3.00 / 1_000_000   # $3.00 per 1M tokens
