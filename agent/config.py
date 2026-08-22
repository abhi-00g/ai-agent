"""
Configuration for the AI Agent.

All settings live here so they're easy to find and change.
In later phases, some of these will come from environment variables
or a YAML config file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM Settings ---
# We use Gemini 2.5 Flash because it's free, fast, and good at following
# structured output formats (which we need for tool call parsing).
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

# --- Agent Settings ---
# Max steps prevents infinite loops. If the agent calls tools 10 times
# without reaching a final answer, something is wrong — force it to stop.
MAX_STEPS = 10

# --- Tool Call Format ---
# This is the format we tell the LLM to use when it wants to call a tool.
# We chose a simple text format over JSON because it's easier to parse
# and Gemini follows it reliably. The format is:
#   TOOL_CALL: tool_name | INPUT: the input to the tool
# When the agent has a final answer and doesn't need any more tools:
#   FINAL_ANSWER: the answer text
TOOL_CALL_PREFIX = "TOOL_CALL:"
TOOL_INPUT_PREFIX = "INPUT:"
FINAL_ANSWER_PREFIX = "FINAL_ANSWER:"
