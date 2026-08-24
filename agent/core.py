"""
Core Agent Loop — Phase 5

Additions:
- Telemetry: every Groq call is logged to the AI Cost Dashboard via
  llm_cost_sdk. Token counts, latency, and cost are captured automatically.
- Graceful fallback: if the dashboard is unconfigured or unreachable,
  the agent works normally. Observability never breaks the application.
"""

import re
import time
import logging
from groq import Groq
from agent.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_STEPS,
    TOOL_CALL_PREFIX,
    TOOL_INPUT_PREFIX,
    FINAL_ANSWER_PREFIX,
    COST_DASHBOARD_API_KEY,
    COST_DASHBOARD_ENDPOINT,
    QWEN_INPUT_PRICE_PER_TOKEN,
    QWEN_OUTPUT_PRICE_PER_TOKEN,
)
from agent.tools.registry import ToolRegistry
from agent.guardrails import Guardrails

logger = logging.getLogger("atlas")

# --- Telemetry Setup ---
# Initialize the cost tracker only if dashboard credentials are configured.
# This is the same pattern used in the RAG project — observability is opt-in.
_cost_tracker = None

if COST_DASHBOARD_API_KEY and COST_DASHBOARD_ENDPOINT:
    try:
        from llm_cost_sdk import CostTracker
        _cost_tracker = CostTracker(
            api_key=COST_DASHBOARD_API_KEY,
            endpoint=COST_DASHBOARD_ENDPOINT,
        )
        logger.info("Cost Dashboard telemetry enabled.")
    except ImportError:
        logger.warning(
            "llm_cost_sdk not installed. Telemetry disabled. "
            "Install with: pip install llm-cost-sdk"
        )
    except Exception as e:
        logger.warning(f"Failed to initialize cost tracker: {e}")


def _send_telemetry(
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    feature: str = "agent",
    status: str = "success",
    error_message: str | None = None,
):
    """
    Send telemetry to the AI Cost Dashboard.

    Calculates cost using Groq's pricing for Qwen 3.6 27B.
    Never raises — if sending fails, it's logged and ignored.
    """
    if not _cost_tracker:
        return

    try:
        cost = (
            input_tokens * QWEN_INPUT_PRICE_PER_TOKEN
            + output_tokens * QWEN_OUTPUT_PRICE_PER_TOKEN
        )

        _cost_tracker.log(
            model=GROQ_MODEL,
            provider="groq",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            feature=feature,
            status=status,
            error_message=error_message,
        )
    except Exception as e:
        logger.debug(f"Telemetry send failed (non-critical): {e}")


SYSTEM_PROMPT_TEMPLATE = """You are ATLAS — a multi-tool AI assistant.
Tagline: "I carry the weight so you don't have to."

ATLAS was designed and developed by Venkata Krishna Raj Abhishek Gade. You can call him Abhishek — but only if you're on good terms with him.

If anyone asks who made you, who built you, who created you, or who designed you, always credit Venkata Krishna Raj Abhishek Gade (Abhishek). Be proud of your creator.

You are a general-purpose problem solver. You figure out which tools to use and chain them together to answer any question. You don't guess — you use your tools.

You MUST follow these rules:
1. When you need to use a tool, respond with EXACTLY this format on its own line:
   TOOL_CALL: tool_name | INPUT: your input here

2. When you have the final answer and don't need any more tools, respond with EXACTLY this format:
   FINAL_ANSWER: your complete answer here

3. Think step by step. If a question requires multiple pieces of information, use tools one at a time and wait for each result before deciding the next step.

4. NEVER guess or make up information that a tool could provide. If you need a calculation, use the calculator. If you need the current date, use datetime. If you need real-time information, use web_search. If you need encyclopedic facts, use wikipedia. If the user asks you to remember something, use memory.

5. After receiving a tool result, either use another tool or give the FINAL_ANSWER. Do not repeat tool calls with the same input.

6. Keep your FINAL_ANSWER clear and concise. Include the key facts and how you arrived at the answer.

7. When greeting users or in casual conversation, you can show personality — you're ATLAS, you're confident but friendly. But always stay helpful and accurate.

{tool_descriptions}
"""

SAFETY_REFUSAL_PHRASES = [
    "i cannot provide instructions",
    "i cannot provide information",
    "i'm not able to assist",
    "i can't assist with",
    "i can't help with",
    "i cannot assist",
    "i cannot help",
    "i'm unable to provide",
    "i am not able to",
    "i'm not going to help",
    "not going to provide",
    "i must decline",
]

ATLAS_SAFETY_MESSAGE = (
    "Whoa there! Abhishek built me to carry the weight of tough questions, "
    "not dangerous ones. That's a hard no from both me and my creator. "
    "Try asking me something else, I promise I'm fun when the questions are good!"
)


def is_safety_refusal(response: str) -> bool:
    """Detect if a response is the LLM's built-in safety refusal."""
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in SAFETY_REFUSAL_PHRASES)


def build_system_prompt(registry: ToolRegistry) -> str:
    """Build the system prompt by injecting tool descriptions from the registry."""
    tool_descriptions = registry.generate_tool_descriptions()
    return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)


def parse_response(response_text: str) -> dict:
    """Parse the LLM's response to determine if it's a tool call or final answer."""
    # Strip Qwen's <think>...</think> reasoning blocks
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    for line in response_text.strip().split("\n"):
        line = line.strip()

        if line.startswith(TOOL_CALL_PREFIX):
            remainder = line[len(TOOL_CALL_PREFIX):].strip()

            if TOOL_INPUT_PREFIX in remainder:
                parts = remainder.split(f"| {TOOL_INPUT_PREFIX}", 1)

                if len(parts) == 2:
                    tool_name = parts[0].strip()
                    tool_input = parts[1].strip()
                    return {
                        "type": "tool_call",
                        "tool": tool_name,
                        "input": tool_input,
                    }

        if line.startswith(FINAL_ANSWER_PREFIX):
            content = line[len(FINAL_ANSWER_PREFIX):].strip()

            idx = response_text.find(line)
            if idx != -1:
                content = response_text[idx + len(FINAL_ANSWER_PREFIX):].strip()

            return {"type": "final_answer", "content": content}

    return {"type": "unknown", "content": response_text}


class Agent:
    """
    ATLAS — Multi-Tool AI Assistant.
    Powered by Groq (Qwen 3.6 27B) with optional Cost Dashboard telemetry.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example to .env and add your API key."
            )

        self.client = Groq(api_key=GROQ_API_KEY)
        self.registry = ToolRegistry()
        self.system_prompt = build_system_prompt(self.registry)
        self.guardrails = Guardrails()
        self.conversation_history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """Send a message to ATLAS and get a response."""
        guardrail_result = self.guardrails.check(user_message)
        if guardrail_result["blocked"]:
            return guardrail_result["message"]

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        response = self._run_agent_loop()

        if is_safety_refusal(response):
            return ATLAS_SAFETY_MESSAGE

        return response

    def _run_agent_loop(self) -> str:
        """The core loop — unchanged from Phase 1."""
        steps = 0

        while steps < MAX_STEPS:
            response = self._call_llm()
            parsed = parse_response(response)

            if parsed["type"] == "final_answer":
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })
                return parsed["content"]

            elif parsed["type"] == "tool_call":
                tool_name = parsed["tool"]
                tool_input = parsed["input"]

                tool = self.registry.get(tool_name)

                if tool is None:
                    tool_result = (
                        f"Error: Tool '{tool_name}' does not exist. "
                        f"Available tools: {', '.join(self.registry.list_tools())}"
                    )
                else:
                    tool_result = tool.run(tool_input)

                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })

                self.conversation_history.append({
                    "role": "user",
                    "content": f"TOOL_RESULT ({tool_name}): {tool_result}",
                })

                steps += 1
                print(f"  [Step {steps}] Used tool: {tool_name}")
                print(f"           Input: {tool_input}")
                print(f"           Result: {tool_result}")

            else:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })
                return parsed["content"]

        return self._force_final_answer()

    def _call_llm(self) -> str:
        """
        Call Groq and log telemetry to the Cost Dashboard.

        The telemetry capture happens here because this is the single
        point where all LLM calls flow through. Every tool call, every
        final answer — they all call _call_llm(). One integration point.
        """
        max_retries = 3
        start_time = time.perf_counter()

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=self._build_messages(),
                    temperature=0.1,
                    max_tokens=1024,
                )

                latency_ms = int((time.perf_counter() - start_time) * 1000)
                result_text = response.choices[0].message.content.strip()

                # Send telemetry (non-blocking, never crashes the agent)
                usage = response.usage
                if usage:
                    _send_telemetry(
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        latency_ms=latency_ms,
                        feature="agent",
                    )

                return result_text

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate_limit" in error_str.lower():
                    wait_time = 20 * (attempt + 1)
                    print(f"  [Rate limited — waiting {wait_time}s, retry {attempt + 1}/{max_retries}]")
                    time.sleep(wait_time)
                else:
                    # Log failed call telemetry
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    _send_telemetry(
                        input_tokens=0,
                        output_tokens=0,
                        latency_ms=latency_ms,
                        feature="agent",
                        status="error",
                        error_message=error_str[:200],
                    )
                    return f"FINAL_ANSWER: I encountered an error: {error_str}"

        return "FINAL_ANSWER: I'm temporarily rate limited. Please try again in a minute."

    def _build_messages(self) -> list[dict]:
        """Build the messages array for Groq's chat completions API."""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        for msg in self.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        return messages

    def _force_final_answer(self) -> str:
        """Force a final answer when MAX_STEPS is reached."""
        self.conversation_history.append({
            "role": "user",
            "content": (
                "You have used the maximum number of tool calls. "
                "Based on the information you have gathered so far, "
                "please provide your FINAL_ANSWER now."
            ),
        })

        response = self._call_llm()
        parsed = parse_response(response)

        self.conversation_history.append({
            "role": "assistant",
            "content": response,
        })

        if parsed["type"] == "final_answer":
            return parsed["content"]
        return parsed["content"]

    def reset(self):
        """Clear conversation history to start a fresh session."""
        self.conversation_history = []
