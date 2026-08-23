"""
Configuration for the AI Agent.

Switched from Google Gemini to Groq in Phase 4.
Why? Gemini's free tier caps at 20 RPD on some accounts, making it
impossible to run a 28-test eval suite. Groq offers 30 RPM and
14,400 RPD on the free tier — no credit card required.

Groq runs open-source models (Llama, Mixtral) on custom LPU hardware,
delivering sub-second inference. We use Llama 3.3 70B which is strong
at structured output and tool-calling patterns.
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