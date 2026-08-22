"""
Tool Registry — Updated for Phase 2

Added WebSearchTool and WikipediaTool to the default registrations.
That's it — two lines added. The agent loop doesn't change at all.
This is the power of the registry + Strategy pattern.
"""

from agent.tools.base import BaseTool
from agent.tools.calculator import CalculatorTool
from agent.tools.datetime_tool import DateTimeTool
from agent.tools.web_search import WebSearchTool
from agent.tools.wikipedia import WikipediaTool


class ToolRegistry:
    """
    Manages all available tools for the agent.

    Usage:
        registry = ToolRegistry()
        tool = registry.get("calculator")
        result = tool.run("2 + 2")
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register all built-in tools. New tools get added here."""
        self.register(CalculatorTool())
        self.register(DateTimeTool())
        self.register(WebSearchTool())
        self.register(WikipediaTool())
        # Phase 3 will add:
        # self.register(MemoryTool())

    def register(self, tool: BaseTool):
        """
        Add a tool to the registry.

        Raises ValueError if a tool with the same name is already registered.
        This prevents silent overwrites — if two tools accidentally share a
        name, you want to know immediately, not debug mysterious behavior later.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"Tool '{tool.name}' is already registered. "
                "Each tool must have a unique name."
            )
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def generate_tool_descriptions(self) -> str:
        """
        Generate the tool description block for the system prompt.

        This is what the LLM sees, so the format matters. Each tool is listed
        with its name and description, making it clear to the LLM what's
        available and when to use each tool.

        Output looks like:
            Available Tools:
            - calculator: Use this tool to evaluate mathematical expressions...
            - datetime: Use this tool to get the current date or time...
            - web_search: Use this tool to search the web...
            - wikipedia: Use this tool to look up factual information...
        """
        if not self._tools:
            return "No tools available."

        lines = ["Available Tools:"]
        for name, tool in self._tools.items():
            lines.append(f"  - {name}: {tool.description}")

        return "\n".join(lines)
