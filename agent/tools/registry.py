"""
Tool Registry

This is the central place where all tools are registered. The registry does
two things:

1. Stores a mapping of tool_name → tool_object, so the agent loop can look
   up and run any tool by name.

2. Generates the "Available Tools" section of the system prompt, so the LLM
   knows what tools exist and when to use each one.

Why a registry instead of a hardcoded list?
Because it's extensible. In Phase 2 when we add web_search and wikipedia,
we just import them here and add one line. The agent loop doesn't change at
all — it already knows how to look up tools by name from the registry.

In interview terms: "The registry decouples tool discovery from tool execution.
Adding a new tool is a one-line change in the registry — zero changes to the
agent loop."
"""

from agent.tools.base import BaseTool
from agent.tools.calculator import CalculatorTool
from agent.tools.datetime_tool import DateTimeTool


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
        # Phase 2 will add:
        # self.register(WebSearchTool())
        # self.register(WikipediaTool())
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
        """
        if not self._tools:
            return "No tools available."

        lines = ["Available Tools:"]
        for name, tool in self._tools.items():
            lines.append(f"  - {name}: {tool.description}")

        return "\n".join(lines)
