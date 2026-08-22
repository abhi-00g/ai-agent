"""
Wikipedia Tool

Fetches article summaries from Wikipedia's free REST API. No API key needed.

Why a separate Wikipedia tool when we already have web search?
1. Wikipedia is free with no rate limits — saves Tavily credits
2. Wikipedia content is structured and factual — better for encyclopedic
   questions than web search results which may include ads, opinions, etc.
3. It teaches the LLM to choose the RIGHT tool: "what is photosynthesis?"
   should use Wikipedia (factual definition), not web search (which would
   return blog posts and study guides)
4. In interviews, having multiple tools that could answer the same question
   demonstrates that your agent makes intelligent tool selection decisions

The API we use:
  https://en.wikipedia.org/api/rest_v1/page/summary/{title}
  Returns a JSON object with the article title, extract (summary), and
  thumbnail image URL. We only use the title and extract.

Error handling:
- Article not found → tells the LLM, which can try web_search instead
- Disambiguation page → tells the LLM to be more specific
- Network error → returns error message, agent continues
"""

import requests
from agent.tools.base import BaseTool


class WikipediaTool(BaseTool):
    """
    Tool for looking up information on Wikipedia.

    Usage by the LLM:
        TOOL_CALL: wikipedia | INPUT: photosynthesis
        TOOL_CALL: wikipedia | INPUT: Eiffel Tower
    """

    BASE_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

    # Timeout for Wikipedia API calls
    REQUEST_TIMEOUT = 10

    @property
    def name(self) -> str:
        return "wikipedia"

    @property
    def description(self) -> str:
        return (
            "Use this tool to look up factual, encyclopedic information from "
            "Wikipedia. Good for: definitions, historical facts, scientific "
            "concepts, biographies, geography, and general knowledge. "
            "Send the topic name like 'photosynthesis' or 'Eiffel Tower'. "
            "Do NOT use this for current events or real-time data — use "
            "web_search for that instead."
        )

    def run(self, tool_input: str) -> str:
        topic = tool_input.strip()

        if not topic:
            return "Error: Please provide a topic to look up on Wikipedia."

        try:
            # Wikipedia's API expects the topic in the URL path.
            # Spaces are automatically handled by requests.
            # We use redirect=true so "USA" redirects to "United States".
            url = f"{self.BASE_URL}/{topic}"

            response = requests.get(
                url,
                params={"redirect": "true"},
                headers={
                    # Wikipedia asks that API users identify themselves
                    # with a User-Agent header. This is good practice.
                    "User-Agent": "AIAgent/1.0 (Educational Project)"
                },
                timeout=self.REQUEST_TIMEOUT,
            )

            # 404 = article not found
            if response.status_code == 404:
                return (
                    f"No Wikipedia article found for '{topic}'. "
                    "Try a different spelling or a more specific/general term."
                )

            response.raise_for_status()

            data = response.json()

            return self._format_result(data)

        except requests.exceptions.Timeout:
            return (
                "Error: Wikipedia request timed out. "
                "The service may be temporarily slow."
            )
        except requests.exceptions.ConnectionError:
            return (
                "Error: Could not connect to Wikipedia. "
                "Check your internet connection."
            )
        except Exception as e:
            return f"Error looking up Wikipedia: {str(e)}"

    def _format_result(self, data: dict) -> str:
        """
        Format Wikipedia's API response into a clean string for the LLM.

        Wikipedia's summary API returns:
        - "title": the canonical article title
        - "extract": a plain-text summary (usually 1-3 paragraphs)
        - "type": "standard" for normal articles, "disambiguation" for
          disambiguation pages, "no-extract" for pages without summaries
        - "description": a short one-line description
        """
        article_type = data.get("type", "")

        # Disambiguation pages list multiple possible meanings
        if article_type == "disambiguation":
            title = data.get("title", "")
            return (
                f"'{title}' is ambiguous and has multiple meanings on Wikipedia. "
                "Please be more specific. For example, if you searched 'Python', "
                "try 'Python programming language' or 'Python snake'."
            )

        title = data.get("title", "Unknown")
        description = data.get("description", "")
        extract = data.get("extract", "No summary available.")

        # Build the response
        parts = [f"Wikipedia: {title}"]

        if description:
            parts.append(f"({description})")

        parts.append(f"\n{extract}")

        return "\n".join(parts)
