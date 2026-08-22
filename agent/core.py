"""
Core Agent Loop

This is the heart of the project. The agent loop:
1. Sends the user's question (with conversation history) to Gemini
2. Reads Gemini's response
3. If Gemini wants a tool → parse, run the tool, feed result back → repeat
4. If Gemini gives a final answer → return it to the user

The loop continues until Gemini gives a FINAL_ANSWER or we hit MAX_STEPS.

Key design decisions:
- Conversation history is a list of dicts with "role" and "content" keys,
  matching the format Gemini expects. This means the LLM sees all previous
  tool calls and results, so it can reference earlier data without re-searching.
- Tool results are added as "user" role messages (the LLM's perspective is:
  "I asked for a tool, and the system gave me the result").
- The system prompt is built once per agent instance and includes all tool
  descriptions from the registry.
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


# --- System Prompt ---
# This is what shapes the agent's behavior. It tells the LLM:
# 1. What it is (a tool-using agent)
# 2. What tools are available (injected from the registry)
# 3. The EXACT format to use for tool calls and final answers
# 4. Rules for good behavior (think step by step, don't guess, etc.)
#
# The format instructions are critical. If the LLM doesn't follow the exact
# format, our parser won't detect the tool call, and the loop breaks.

SYSTEM_PROMPT_TEMPLATE = """You are an intelligent AI assistant that can use tools to answer questions.

You MUST follow these rules:
1. When you need to use a tool, respond with EXACTLY this format on its own line:
   TOOL_CALL: tool_name | INPUT: your input here

2. When you have the final answer and don't need any more tools, respond with EXACTLY this format:
   FINAL_ANSWER: your complete answer here

3. Think step by step. If a question requires multiple pieces of information, use tools one at a time and wait for each result before deciding the next step.

4. NEVER guess or make up information that a tool could provide. If you need a calculation, use the calculator. If you need the current date, use datetime.

5. After receiving a tool result, either use another tool or give the FINAL_ANSWER. Do not repeat tool calls with the same input.

6. Keep your FINAL_ANSWER clear and concise. Include the key facts and how you arrived at the answer.

{tool_descriptions}
"""


def build_system_prompt(registry: ToolRegistry) -> str:
    """
    Build the system prompt by injecting tool descriptions from the registry.

    This is why the registry exists — when you add a new tool in Phase 2,
    the system prompt automatically includes it. You never manually edit
    the prompt to add tool descriptions.
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

    Why return a dict instead of a tuple?
    Because dicts are self-documenting. When you read parse_result["type"],
    you know exactly what you're checking. A tuple like (True, "calculator",
    "245 * 18") requires you to remember what each position means.
    """
    # Check each line — the tool call or final answer might not be on the
    # first line (the LLM sometimes adds thinking text before it)
    for line in response_text.strip().split("\n"):
        line = line.strip()

        # Check for a tool call: "TOOL_CALL: calculator | INPUT: 245 * 18"
        if line.startswith(TOOL_CALL_PREFIX):
            remainder = line[len(TOOL_CALL_PREFIX):].strip()

            if TOOL_INPUT_PREFIX in remainder:
                # Split on " | INPUT: " to get tool name and input
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

            # Sometimes the final answer spans multiple lines after the prefix.
            # Find where in the original text this line starts and take
            # everything after the prefix.
            idx = response_text.find(line)
            if idx != -1:
                content = response_text[idx + len(FINAL_ANSWER_PREFIX):].strip()

            return {"type": "final_answer", "content": content}

    # If we get here, the LLM didn't follow the format. This shouldn't
    # happen often with a good system prompt, but we handle it gracefully.
    return {"type": "unknown", "content": response_text}


class Agent:
    """
    The main agent class. Create an instance and call chat() to interact.

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

        # Build the system prompt with tool descriptions
        self.system_prompt = build_system_prompt(self.registry)

        # Conversation history persists across chat() calls within the same
        # session. This is how the agent "remembers" within a conversation.
        # When the Python process restarts, this is gone — that's what the
        # memory tool in Phase 3 will fix.
        self.conversation_history: list[dict] = []

    def chat(self, user_message: str) -> str:
        """
        Send a message to the agent and get a response.

        This is the public API. The user calls this, and the agent loop
        handles everything internally — tool calls, retries, parsing.
        The user just sees the final answer.

        Args:
            user_message: The user's question or message.

        Returns:
            The agent's final answer as a string.
        """
        # Add the user's message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        # Run the agent loop
        return self._run_agent_loop()

    def _run_agent_loop(self) -> str:
        """
        The core loop. Keeps calling Gemini and running tools until we get
        a final answer or hit the step limit.

        Step tracking:
        - Each tool call counts as one step
        - If we hit MAX_STEPS, we force the agent to summarize what it has
          so far. This prevents infinite loops where the agent keeps calling
          tools without converging on an answer.
        """
        steps = 0

        while steps < MAX_STEPS:
            # --- Call Gemini ---
            response = self._call_llm()

            # --- Parse the response ---
            parsed = parse_response(response)

            if parsed["type"] == "final_answer":
                # The agent is done — save the answer to history and return
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })
                return parsed["content"]

            elif parsed["type"] == "tool_call":
                tool_name = parsed["tool"]
                tool_input = parsed["input"]

                # Look up the tool in the registry
                tool = self.registry.get(tool_name)

                if tool is None:
                    # The LLM hallucinated a tool that doesn't exist.
                    # Tell it the tool doesn't exist so it can try again.
                    tool_result = (
                        f"Error: Tool '{tool_name}' does not exist. "
                        f"Available tools: {', '.join(self.registry.list_tools())}"
                    )
                else:
                    # Run the tool and get the result
                    tool_result = tool.run(tool_input)

                # Save the LLM's tool call to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })

                # Send the tool result back to the LLM.
                # We format it clearly so the LLM knows this is a tool result
                # and not a user message.
                self.conversation_history.append({
                    "role": "user",
                    "content": f"TOOL_RESULT ({tool_name}): {tool_result}",
                })

                steps += 1
                print(f"  [Step {steps}] Used tool: {tool_name}")
                print(f"           Input: {tool_input}")
                print(f"           Result: {tool_result}")

            else:
                # The LLM didn't follow the format. Save what it said and
                # return it as the response. In most cases, this means the
                # LLM just answered directly without using the format — which
                # is fine for simple questions that don't need tools.
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response,
                })
                return parsed["content"]

        # If we get here, we've exhausted MAX_STEPS without a final answer.
        # Ask the LLM to summarize what it has so far.
        return self._force_final_answer()

    def _call_llm(self) -> str:
        """
        Call Gemini with the current conversation history.

        The system prompt is passed via the system_instruction parameter,
        and the conversation history is passed as contents.
        """
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=self._build_contents(),
                config={
                    "system_instruction": self.system_prompt,
                    "temperature": 0.1,  # Low temperature = more deterministic
                    # We want consistent, predictable tool calls, not creative
                    # outputs. Higher temperature makes the LLM more "random"
                    # which would cause inconsistent format adherence.
                },
            )
            return response.text.strip()

        except Exception as e:
            return f"FINAL_ANSWER: I encountered an error communicating with the AI model: {str(e)}"

    def _build_contents(self) -> list[dict]:
        """
        Convert our conversation history into the format Gemini expects.

        Gemini expects a list of Content objects with "role" and "parts".
        Our internal format uses "role" and "content" (simpler to work with).
        This method converts between the two.
        """
        contents = []
        for msg in self.conversation_history:
            contents.append({
                "role": msg["role"] if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}],
            })
        return contents

    def _force_final_answer(self) -> str:
        """
        When the agent hits MAX_STEPS, force it to give a final answer
        based on whatever information it has collected so far.
        """
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
