"""Tests for the tool registry — no API calls needed."""

import pytest
from agent.tools.registry import ToolRegistry
from agent.tools.base import BaseTool


class MockTool(BaseTool):
    """A fake tool for testing the registry."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing."

    def run(self, tool_input: str) -> str:
        return f"mock result: {tool_input}"


class TestToolRegistry:
    def test_default_tools_registered(self):
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert "calculator" in tools
        assert "datetime" in tools
        assert "web_search" in tools
        assert "wikipedia" in tools
        assert "memory" in tools

    def test_get_existing_tool(self):
        registry = ToolRegistry()
        tool = registry.get("calculator")
        assert tool is not None
        assert tool.name == "calculator"

    def test_get_nonexistent_tool(self):
        registry = ToolRegistry()
        tool = registry.get("nonexistent")
        assert tool is None

    def test_register_custom_tool(self):
        registry = ToolRegistry()
        mock = MockTool()
        registry.register(mock)
        assert "mock_tool" in registry.list_tools()

    def test_reject_duplicate_name(self):
        registry = ToolRegistry()
        mock = MockTool()
        registry.register(mock)
        with pytest.raises(ValueError):
            registry.register(mock)

    def test_generate_descriptions(self):
        registry = ToolRegistry()
        desc = registry.generate_tool_descriptions()
        assert "Available Tools:" in desc
        assert "calculator" in desc
        assert "datetime" in desc

    def test_tool_count(self):
        registry = ToolRegistry()
        assert len(registry.list_tools()) == 5
