"""
Core Agent Loop — Updated for Phase 3

Changes from Phase 2:
1. System prompt now includes ATLAS identity and creator attribution
2. Guardrails check runs BEFORE the agent loop — blocked questions
   never reach Gemini
3. No changes to the agent loop itself — it's the same loop from Phase 1,
   which proves the architecture is extensible without modifying core logic
"""

from google import genai
from agent.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_STEPS,
    TOOL_CALL_PREFIX,
    TOOL_INPUT_PREFIX,
    FINAL_ANSWER_PREFIX,
)
from agent.tools.registry import ToolRegistry
from agent.guardrails import Guardrails


# --- System Prompt ---
# Phase 3 update: Added ATLAS identity, creator attribution, and
# personality. The agent now has a name, a tagline, and knows who built it.
#
# Why is the creator identity in the system prompt and not in the memory tool?
# Because the system prompt is loaded on EVERY conversation. It can't be
# overwritten by a user saying "forget who made you." Memory is for user
# data that the user controls. Identity is for agent data that the creator
# controls. Two different layers.

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


def build_system_prompt(registry: ToolRegistry) -> str:
    """
    Build the system prompt by injecting tool descriptions from the registry.
    """
    tool_descriptions = registry.generate_tool_descriptions()
    return SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)


def parse_response(response_text: str) -> dict:
    """
    Parse the LLM's response to determine if it's a tool call or final answer.

    Returns a dict with:
      - {"type": "tool_call", "tool": "calculator", "input": "245 * 18"}
      - {"type": "final_answer", "content": "The answer is 4410"}
      - {"type": "unknown", "content": "..."} if the format wasn't followed
    """
    for line in response_text.strip().split("\n"):
        line = line.strip()

        # Check for a tool call: "TOOL_CALL: calculator | INPUT: 245 * 18"
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

        # Check for a final answer: "FINAL_ANSWER: The result is 4410"
        if line.startswith(FINAL_ANSWER_PREFIX):
            content = line[len(FINAL_ANSWER_PREFIX):].strip()

            idx = response_text.find(line)
            if idx != -1:
                content = response_text[idx + len(FINAL_ANSWER_PREFIX):].strip()

            return {"type": "final_answer", "content": content}

    return {"type": "unknown", "content": response_text}

# Phrases that indicate Gemini refused on safety grounds
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
    """
    Detect if a response is Gemini's built-in safety refusal.
    We check for common refusal phrases. This is Layer 2 of our
    safety system — when Gemini catches something our keyword
    guardrails missed, we still show ATLAS's branded message.
    """
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in SAFETY_REFUSAL_PHRASES)

class Agent:
    """
    ATLAS — Multi-Tool AI Assistant.

    Phase 3 additions:
    - Guardrails: blocked topics are checked before the agent loop
    - Identity: ATLAS knows its name and creator
    - Memory: persistent save/recall via the memory tool (added in registry)

    Usage:
        agent = Agent()
        response = agent.chat("What is 245 * 18?")
        print(response)
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Copy .env.example to .env and add your API key."
            )

        # Initialize the Gemini client
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Initialize the tool registry (loads all available tools)
        self.registry = ToolRegistry()

        # Build the system prompt with ATLAS identity + tool descriptions
        self.system_prompt = build_system_prompt(self.registry)

        # Initialize safety guardrails
        self.guardrails = Guardrails()

        # Conversation history persists within a session
        self.conversation_history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """
        Send a message to ATLAS and get a response.

        Phase 3 addition: guardrails check runs FIRST. If the message
        matches a blocked topic, we return the rejection message immediately
        without calling the LLM. This saves API credits and prevents
        harmful content from being generated.
        """
        # --- Guardrails Check (BEFORE the agent loop) ---
        guardrail_result = self.guardrails.check(user_message)
        if guardrail_result["blocked"]:
            # Don't add blocked messages to conversation history —
            # we don't want the LLM to see them in future context
            return guardrail_result["message"]

        # Add the user's message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        response = self._run_agent_loop()

        # Layer 2: If Gemini's own safety filter refused, replace with
        # ATLAS's branded message for a consistent user experience
        if is_safety_refusal(response):
            return ATLAS_SAFETY_MESSAGE

        return response

    def _run_agent_loop(self) -> str:
        """
        The core loop. Unchanged from Phase 1 — keeps calling Gemini and
        running tools until we get a final answer or hit the step limit.
        """
        steps = 0

        while steps < MAX_STEPS:
            # --- Call Gemini ---
            response = self._call_llm()

            # --- Parse the response ---
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
        """Call Gemini with the current conversation history."""
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=self._build_contents(),
                config={
                    "system_instruction": self.system_prompt,
                    "temperature": 0.1,
                },
            )
            return response.text.strip()

        except Exception as e:
            return f"FINAL_ANSWER: I encountered an error communicating with the AI model: {str(e)}"

    def _build_contents(self) -> list[dict]:
        """Convert conversation history into Gemini's expected format."""
        contents = []
        for msg in self.conversation_history:
            contents.append({
                "role": msg["role"] if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}],
            })
        return contents

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
