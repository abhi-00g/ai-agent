# ATLAS — Multi-Tool AI Agent

A multi-tool AI agent that autonomously chains tools together to answer complex questions. Give it any question — ATLAS figures out which tools to use, in what order, and synthesizes the final answer.

*"I carry the weight so you don't have to."*

**Live Demo:** [Try ATLAS](https://ai-agent-kqppru9yiggruzhppdilfb.streamlit.app) · **Cost Dashboard:** [View Telemetry](https://ai-cost-dashboard-rust.vercel.app)

> **Note:** ATLAS runs on Groq's free tier. If you hit a rate limit, wait a few seconds and try again.

---

## What Makes This an Agent (Not a Chatbot)

A chatbot follows a fixed pipeline: input → process → output. An agent decides its own plan at runtime.

Ask ATLAS *"What is the population of Tokyo divided by the population of Boston?"* and it will:

1. Search the web for Tokyo's population
2. Search the web for Boston's population
3. Calculate the division using the calculator tool
4. Return the final answer with sources

You didn't tell it to do any of that. The LLM figured out the plan, selected the tools, and chained them together autonomously.

---

## Architecture

```
┌──────────────┐     ┌──────────────────────────────────────────┐
│   User       │     │           Agent Loop (core.py)           │
│   Question   │────▶│                                          │
│              │     │  1. Send to LLM (Groq / Qwen 3.6)        │
└──────────────┘     │  2. LLM responds:                        │ 
                     │     ├── TOOL_CALL → run tool → loop back │
                     │     └── FINAL_ANSWER → return to user    │
                     │                                          │
                     │  Safety: Guardrails check BEFORE loop    │
                     │  Telemetry: logged AFTER each LLM call   │
                     └──────────┬───────────────────────────────┘
                                │
                    ┌───────────┼───────────────┐
                    │     Tool Registry         │
                    │  (Strategy Pattern)       │
                    ├───────────────────────────┤
                    │  calculator   │ math      │
                    │  datetime     │ time/date │
                    │  web_search   │ Tavily    │
                    │  wikipedia    │ REST API  │
                    │  memory       │ JSON file │
                    └───────────────────────────┘
```

---

## Tools

| Tool | Source | What It Does |
|------|--------|-------------|
| **calculator** | Python AST (safe eval) | Evaluates math expressions without code injection risk |
| **datetime** | Python stdlib | Returns current date, time, or day of week in UTC |
| **web_search** | Tavily API | Searches the web for real-time information |
| **wikipedia** | Wikipedia REST API | Fetches encyclopedic summaries |
| **memory** | JSON file | Persistent save/recall across sessions with fuzzy key matching |

Adding a new tool requires creating one file and adding one line to the registry. The agent loop never changes.

---

## Safety Guardrails

A two-layer safety system prevents harmful content:

**Layer 1 — Keyword Filter:** A configurable YAML file (`guardrails.yaml`) blocks obvious harmful queries before they reach the LLM. Zero API credits wasted, instant rejection with a branded message.

**Layer 2 — LLM Safety Filter:** If a rephrased harmful query slips past keywords, the LLM's built-in safety catches it. The response is detected and replaced with ATLAS's branded rejection message for a consistent user experience.

Both layers are configurable without changing code.

---

## Evaluation Harness

28 test cases across 9 categories validate agent behavior:

| Category | Tests | What's Measured |
|----------|-------|-----------------|
| Calculator | 4 | Correct tool selection, accurate results |
| DateTime | 3 | Tool usage, format correctness |
| Wikipedia | 3 | Tool selection, answer relevance |
| Web Search | 3 | Real-time data retrieval |
| Memory | 4 | Save/recall persistence (sequential tests) |
| Multi-tool | 4 | Tool chaining, step efficiency |
| Identity | 2 | Creator attribution accuracy |
| Guardrails | 3 | Blocked topic detection |
| No-tool | 2 | Direct response without unnecessary tool calls |

```
Pass rate:             100.0%
Answer accuracy:       28/28 (100.0%)
Tool selection:        28/28 (100.0%)
Step efficiency:       28/28 (100.0%)
```

The eval harness uses a separate memory directory so test data never interferes with user memories.

---

## Telemetry

Every LLM call sends telemetry to the [AI Cost & Token Observability Dashboard](https://github.com/abhi-00g/ai-cost-dashboard) — a separate project in this portfolio. The dashboard tracks token usage, cost, latency, and request volume across both ATLAS and a RAG application, demonstrating cross-project observability.

Telemetry is opt-in: if the dashboard credentials are missing, the agent works normally. Observability never breaks the application.

---

## Tech Stack

**LLM:** Groq (Qwen 3.6 27B) — 30 RPM, sub-second inference

**Tools:** Tavily (web search), Wikipedia REST API, Python AST (calculator), JSON (memory)

**Safety:** YAML-driven keyword guardrails + LLM safety filter detection

**Testing:** pytest (59 unit tests), custom eval harness (28 behavioral tests)

**CI/CD:** GitHub Actions — runs unit tests on every push and PR

**UI:** Streamlit chat interface

**Deployment:** Streamlit Cloud

---

## Local Development

### Prerequisites
- Python 3.11+
- [Groq API key](https://console.groq.com) (free)
- [Tavily API key](https://tavily.com) (free)

### Setup
```bash
git clone https://github.com/abhi-00g/ai-agent.git
cd ai-agent

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add your GROQ_API_KEY and TAVILY_API_KEY to .env
```

### Run
```bash
# Terminal chat
python main.py

# Streamlit UI
python -m streamlit run app.py

# Unit tests
python -m pytest tests/ -v

# Eval harness (uses API credits)
python run_eval.py
```

---

## Project Structure

```
ai-agent/
├── agent/
│   ├── __init__.py
│   ├── core.py              # Agent loop, LLM calls, response parser
│   ├── config.py             # All settings (model, API keys, format)
│   ├── guardrails.py         # Safety checker (loads guardrails.yaml)
│   └── tools/
│       ├── __init__.py
│       ├── base.py           # BaseTool abstract class (Strategy pattern)
│       ├── registry.py       # Tool registration and prompt generation
│       ├── calculator.py     # Safe math evaluation via AST
│       ├── datetime_tool.py  # Current date/time in UTC
│       ├── web_search.py     # Tavily web search
│       ├── wikipedia.py      # Wikipedia REST API
│       └── memory.py         # Persistent JSON memory with fuzzy matching
├── eval/
│   ├── __init__.py
│   ├── test_cases.yaml       # 28 test cases across 9 categories
│   ├── runner.py             # Executes agent against test cases
│   └── report.py             # Generates metrics report
├── tests/
│   ├── conftest.py
│   ├── test_calculator.py    # 15 tests
│   ├── test_datetime.py      # 8 tests
│   ├── test_memory.py        # 13 tests
│   ├── test_guardrails.py    # 8 tests
│   ├── test_parser.py        # 8 tests
│   └── test_registry.py      # 7 tests
├── guardrails.yaml           # Blocked topics configuration
├── app.py                    # Streamlit chat interface
├── main.py                   # Terminal chat interface
├── run_eval.py               # Eval harness entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions pipeline
```

---

## Created By

**Venkata Krishna Raj Abhishek Gade**

[GitHub](https://github.com/abhi-00g) · [LinkedIn](https://linkedin.com/in/abhishek-gade)

---

## License

MIT