"""
Tool Registry — Updated for Phase 3

Added MemoryTool. One new import, one new register line.
The pattern continues: adding a tool never touches core.py.
"""

from agent.tools.base import BaseTool
from agent.tools.calculator import CalculatorTool
from agent.tools.datetime_tool import DateTimeTool
from agent.tools.web_search import WebSearchTool
from agent.tools.wikipedia import WikipediaTool
from agent.tools.memory import MemoryTool


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
        self.register(MemoryTool())

    def register(self, tool: BaseTool):
        """
        Add a tool to the registry.

        Raises ValueError if a tool with the same name is already registered.
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
        """
        if not self._tools:
            return "No tools available."

        lines = ["Available Tools:"]
        for name, tool in self._tools.items():
            lines.append(f"  - {name}: {tool.description}")

        return "\n".join(lines)
