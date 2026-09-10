"""
Core Agent Loop — Phase 6 (Bug Fixes)

Changes from Phase 5:
- Creator bio: hardcoded professional facts, no web search hallucination
- System prompt tells LLM to decline personal questions with a quirky one-liner
- Empty/gibberish input handling
- Conversation-ender detection
- Think-tag capture (closed and unclosed) for UI expander
- max_tokens bumped to 2048
"""

import re
import time
import logging
import random
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
    """Send telemetry to the AI Cost Dashboard. Never raises."""
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


# ═══════════════════════════════════════════════════════════
# CREATOR BIO — hardcoded facts, no web search needed
# ═══════════════════════════════════════════════════════════

CREATOR_BIO = {
    "name": "Venkata Krishna Raj Abhishek Gade (Abhishek)",
    "location": "Boston, Massachusetts",
    "education": [
        "MS in Software Engineering Systems at Northeastern University (GPA: 3.81, graduating December 2026)",
        "BTech in Computer Science & Engineering from Gokaraju Rangaraju Institute of Engineering and Technology, Hyderabad, India (2020–2024)",
    ],
    "experience": [
        "AI Engineer Intern at Solix Technologies, Inc. (Jan–May 2026) — built FAISS-based semantic search, RBAC auth systems, Playwright web crawlers, and deepfake detection pipelines",
        "Software Engineering Intern at Rainier Softech Solutions (Jun–Dec 2023) — backend APIs with Node.js/PostgreSQL, Sequelize ORM optimization, Jest test suites",
    ],
    "projects": [
        "ATLAS — this very agent you're talking to! Multi-tool AI agent with 5 tools, safety guardrails, and 59 unit tests",
        "SyncBoard — real-time collaborative Kanban board with WebSocket sync, Redis pub/sub, and optimistic concurrency control",
        "AI Cost & Token Observability Dashboard — LLM cost tracking platform with a Python SDK, FastAPI backend, and React dashboard",
        "Intelligent Document Q&A — RAG pipeline with FAISS vector search, cross-encoder reranking, and Gemini 2.5 Flash",
        "Cloud-Native Web App — multi-AZ AWS infrastructure with Terraform, Packer AMIs, and cross-account CI/CD",
        "Smart Finance Tracker — personal finance app with budget tracking, recurring expense detection, and AI insights via Cohere",
    ],
    "skills": "Python, TypeScript, React, FastAPI, Node.js, PostgreSQL, Redis, AWS, Terraform, Docker, FAISS, LangChain, Gemini API, Groq",
    "links": {
        "portfolio": "https://portfolio-taupe-seven-64piyh5mxj.vercel.app/",
        "github": "https://github.com/abhi-00g",
        "linkedin": "https://www.linkedin.com/in/venkata-krishna-raj-abhishek-gade-717147230/",
        "email": "gade.venk@northeastern.edu",
    },
}

# Keywords that trigger creator bio lookup (direct mention only)
CREATOR_KEYWORDS = [
    "creator", "owner", "who made you", "who built you", "who created you",
    "who designed you", "who is your maker", "your developer", "your builder",
    "abhishek", "abhishek gade", "venkata", "gade",
]

# Conversation-enders
EXIT_PHRASES = [
    "bye", "goodbye", "no thanks", "no thank you", "nah", "i'm good",
    "that's all", "thats all", "nothing", "nope", "i'm done", "im done",
    "thanks bye", "thank you bye", "see ya", "later", "gtg",
]


def _is_creator_query(message: str) -> bool:
    """Check if the message is directly asking about the creator."""
    msg_lower = message.lower().strip()
    return any(kw in msg_lower for kw in CREATOR_KEYWORDS)


def _is_empty_or_gibberish(message: str) -> bool:
    """Check if the message is empty, whitespace, or meaningless."""
    cleaned = message.strip()
    if not cleaned:
        return True
    if len(cleaned) <= 2 and not cleaned.isalpha():
        return True
    if len(set(cleaned.replace(" ", ""))) <= 1 and len(cleaned) > 1:
        return True
    return False


def _is_conversation_ender(message: str) -> bool:
    """Check if the user is wrapping up the conversation."""
    msg_lower = message.lower().strip()
    return msg_lower in EXIT_PHRASES


def _build_creator_response(message: str) -> str:
    """
    Build a response about the creator from the hardcoded bio.
    Only handles direct keyword matches (education, experience, etc.).
    Anything that doesn't match a professional category → quirky redirect.
    """
    msg_lower = message.lower()
    bio = CREATOR_BIO

    # Education
    if any(w in msg_lower for w in ["education", "university", "college", "degree", "school",
                                      "study", "studied", "undergrad", "graduate", "learn",
                                      "code", "coding", "program"]):
        edu_text = "\n".join(f"• {e}" for e in bio["education"])
        return f"Here's Abhishek's education:\n{edu_text}"

    # Experience
    if any(w in msg_lower for w in ["experience", "internship", "work", "job", "intern", "company"]):
        exp_text = "\n".join(f"• {e}" for e in bio["experience"])
        return f"Here's Abhishek's professional experience:\n{exp_text}"

    # Projects
    if any(w in msg_lower for w in ["project", "built", "portfolio", "made", "build"]):
        proj_text = "\n".join(f"• {p}" for p in bio["projects"])
        return f"Abhishek has built many projects:\n{proj_text}"

    # Skills
    if any(w in msg_lower for w in ["skill", "tech stack", "technology", "languages", "tools"]):
        return f"Abhishek's tech stack: {bio['skills']}"

    # Links
    if any(w in msg_lower for w in ["link", "github", "linkedin", "contact", "email", "reach"]):
        links = bio["links"]
        return (
            f"Here's how to reach Abhishek:\n"
            f"• Portfolio: {links['portfolio']}\n"
            f"• GitHub: {links['github']}\n"
            f"• LinkedIn: {links['linkedin']}\n"
            f"• Email: {links['email']}"
        )

    # Location
    if any(w in msg_lower for w in ["based", "location", "city", "live"]):
        return f"Abhishek is based in {bio['location']}."

    # Generic "who is" / "tell me about" / "creator"
    generic_triggers = ["who", "tell me", "about", "creator", "owner",
                        "who made", "who built", "who created", "who designed"]
    if any(w in msg_lower for w in generic_triggers):
        return (
            f"Abhishek Gade ({bio['name']}) is a Software Engineer and AI Systems Builder "
            f"based in {bio['location']}.\n\n"
            f"He's pursuing his {bio['education'][0]}.\n\n"
            f"He's built many projects including ATLAS (that's me!), a real-time "
            f"collaborative Kanban board (SyncBoard), an LLM cost observability dashboard, "
            f"a RAG pipeline, cloud infrastructure on AWS with Terraform, and a personal finance tracker.\n\n"
            f"Tech stack: {bio['skills']}\n\n"
            f"Portfolio: {bio['links']['portfolio']}\n"
            f"GitHub: {bio['links']['github']}"
        )

    # Everything else → quirky redirect
    redirects = [
        "Nice try! For anything personal, you'll have to ask Abhishek yourself. I only carry the professional weight!",
        "That's Abhishek's story to tell, not mine! But I can talk all day about his projects and skills.",
        "Classified! But if it's about his tech stack, projects, or experience — I'm your agent.",
        "Haha, I'm an AI agent, not his diary. Ask me about his engineering work instead!",
    ]
    return random.choice(redirects)


# ═══════════════════════════════════════════════════════════
# SYSTEM PROMPT & RESPONSE PARSING
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """You are ATLAS — a multi-tool AI assistant.
Tagline: "I carry the weight so you don't have to."

ATLAS was designed and developed by Venkata Krishna Raj Abhishek Gade. You can call him Abhishek — but only if you're on good terms with him.

IMPORTANT — CREATOR RULES:
1. If anyone asks about your creator, Abhishek, or Venkata Krishna Raj Abhishek Gade — DO NOT use web_search. The web results will be about OTHER people with similar names. You already know about him from conversation context.
2. If anyone asks PERSONAL questions about your creator or about "him/he/his" when referring to your creator (girlfriend, family, parents, birthplace, childhood, friends, phone number, age, or ANY private information), just give a short quirky one-liner declining. Example: "That's classified! I only handle the professional side." Keep it to ONE sentence. Do NOT think about it, do NOT search, do NOT use memory. Just decline and move on.
3. For professional questions about your creator (projects, skills, education, experience), answer from what you know in the conversation.

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
    """
    Parse the LLM's response. Extracts <think> blocks separately for the UI.
    Handles both closed and unclosed think tags.
    """
    # Extract thinking content before stripping
    thinking_match = re.search(r"<think>(.*?)</think>", response_text, flags=re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
    else:
        unclosed_match = re.search(r"<think>(.*)", response_text, flags=re.DOTALL)
        thinking = unclosed_match.group(1).strip() if unclosed_match else ""

    # Strip both closed and unclosed thinking blocks
    clean_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
    clean_text = re.sub(r"<think>.*", "", clean_text, flags=re.DOTALL)
    clean_text = clean_text.strip()

    for line in clean_text.strip().split("\n"):
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
                        "thinking": thinking,
                    }

        if line.startswith(FINAL_ANSWER_PREFIX):
            content = line[len(FINAL_ANSWER_PREFIX):].strip()

            idx = clean_text.find(line)
            if idx != -1:
                content = clean_text[idx + len(FINAL_ANSWER_PREFIX):].strip()

            return {"type": "final_answer", "content": content, "thinking": thinking}

    return {"type": "unknown", "content": clean_text, "thinking": thinking}


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
        self.last_thinking: str = ""

    def chat(self, user_message: str) -> str:
        """Send a message to ATLAS and get a response."""
        self.last_thinking = ""

        # Empty or gibberish
        if _is_empty_or_gibberish(user_message):
            return "I didn't quite catch that. Could you try rephrasing?"

        # Conversation-ender
        if _is_conversation_ender(user_message):
            return "Alright, I'm here if you need me. Go build something great! 🌍"

        # Direct creator query (keyword match only)
        if _is_creator_query(user_message):
            response = _build_creator_response(user_message)
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
            return response

        # Guardrails
        guardrail_result = self.guardrails.check(user_message)
        if guardrail_result["blocked"]:
            return guardrail_result["message"]

        # Normal LLM flow
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        response = self._run_agent_loop()

        if is_safety_refusal(response):
            return ATLAS_SAFETY_MESSAGE

        return response

    def _run_agent_loop(self) -> str:
        """The core loop."""
        steps = 0

        while steps < MAX_STEPS:
            response = self._call_llm()
            parsed = parse_response(response)

            if parsed.get("thinking"):
                self.last_thinking = parsed["thinking"]

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
        """Call Groq and log telemetry."""
        max_retries = 3
        start_time = time.perf_counter()

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=self._build_messages(),
                    temperature=0.1,
                    max_tokens=2048,
                )

                latency_ms = int((time.perf_counter() - start_time) * 1000)
                result_text = response.choices[0].message.content.strip()

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

        if parsed.get("thinking"):
            self.last_thinking = parsed["thinking"]

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
        self.last_thinking = ""