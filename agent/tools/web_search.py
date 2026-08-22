"""
Web Search Tool (Tavily)

Searches the web using Tavily's API. Tavily is designed specifically for
AI agents — it returns clean, extracted content (not raw HTML) which makes
it much easier for the LLM to process compared to Google's API.

Why Tavily instead of Google Custom Search or SerpAPI?
1. Free tier: 1,000 searches/month, no credit card required
2. Returns clean text, not HTML snippets — less parsing needed
3. LangChain and OpenAI both use Tavily as their default search tool,
   so it's well-documented and widely adopted in the AI agent ecosystem
4. Has a "basic" mode (1 credit) that returns snippets, perfect for our use

Error handling:
- Network timeout → returns error message, agent can try a different approach
- Invalid API key → returns clear error message
- No results found → tells the LLM so it can rephrase or try Wikipedia
- Any unexpected error → caught and returned as text, never crashes the agent
"""

import os
import requests
from agent.tools.base import BaseTool


class WebSearchTool(BaseTool):
    """
    Tool for searching the web using Tavily's API.

    Usage by the LLM:
        TOOL_CALL: web_search | INPUT: current GDP of France in USD
    """

    TAVILY_API_URL = "https://api.tavily.com/search"

    # Timeout in seconds. If Tavily doesn't respond in 10 seconds,
    # we give up rather than blocking the agent forever.
    REQUEST_TIMEOUT = 10

    # How many search results to return. More results = more context for
    # the LLM, but also more tokens used. 3 is a good balance.
    MAX_RESULTS = 3

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Use this tool to search the web for current, real-time information. "
            "Good for: recent events, current prices, live data, news, "
            "anything that changes over time. "
            "Send a clear search query like 'current population of Japan 2026' "
            "or 'latest iPhone model and price'."
        )

    def run(self, tool_input: str) -> str:
        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            return (
                "Error: TAVILY_API_KEY is not set. "
                "Please add it to your .env file. "
                "Get a free key at https://tavily.com"
            )

        try:
            # Tavily API request
            # search_depth="basic" costs 1 credit (vs "advanced" which
            # costs 2 credits and extracts full page content).
            # For our agent, snippets are enough — the LLM doesn't need
            # full articles to answer most questions.
            payload = {
                "api_key": api_key,
                "query": tool_input.strip(),
                "search_depth": "basic",
                "max_results": self.MAX_RESULTS,
                "include_answer": True,
            }

            response = requests.post(
                self.TAVILY_API_URL,
                json=payload,
                timeout=self.REQUEST_TIMEOUT,
            )

            # Check for HTTP errors (4xx, 5xx)
            response.raise_for_status()

            data = response.json()

            # Build a clean result string for the LLM
            return self._format_results(data)

        except requests.exceptions.Timeout:
            return (
                "Error: Web search timed out. The search service may be "
                "temporarily slow. Try a simpler query or use a different tool."
            )
        except requests.exceptions.ConnectionError:
            return (
                "Error: Could not connect to the search service. "
                "Check your internet connection."
            )
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                return "Error: Invalid Tavily API key. Check your .env file."
            return f"Error: Search request failed with status {e.response.status_code if e.response else 'unknown'}."
        except Exception as e:
            return f"Error performing web search: {str(e)}"

    def _format_results(self, data: dict) -> str:
        """
        Format Tavily's response into a clean string for the LLM.

        Tavily returns:
        - "answer": a direct AI-generated answer (if include_answer=True)
        - "results": list of search results with title, url, and content

        We include both so the LLM has the direct answer plus supporting
        sources it can reference.
        """
        parts = []

        # Include Tavily's direct answer if available
        answer = data.get("answer")
        if answer:
            parts.append(f"Direct answer: {answer}")

        # Include individual search results
        results = data.get("results", [])
        if results:
            parts.append(f"\nSearch results ({len(results)} found):")
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "")

                # Truncate long content to save tokens
                if len(content) > 300:
                    content = content[:300] + "..."

                parts.append(f"\n{i}. {title}")
                parts.append(f"   {content}")
                if url:
                    parts.append(f"   Source: {url}")

        if not parts:
            return "No search results found. Try rephrasing your query."

        return "\n".join(parts)
