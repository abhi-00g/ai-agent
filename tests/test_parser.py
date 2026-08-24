"""Tests for the response parser — no API calls needed."""

from agent.core import parse_response


class TestParseResponse:
    def test_parses_tool_call(self):
        result = parse_response("TOOL_CALL: calculator | INPUT: 245 * 18")
        assert result["type"] == "tool_call"
        assert result["tool"] == "calculator"
        assert result["input"] == "245 * 18"

    def test_parses_final_answer(self):
        result = parse_response("FINAL_ANSWER: The result is 4410")
        assert result["type"] == "final_answer"
        assert "4410" in result["content"]

    def test_parses_tool_call_with_thinking(self):
        """Should strip <think> blocks before parsing."""
        text = "<think>I need to calculate this</think>\nTOOL_CALL: calculator | INPUT: 2 + 2"
        result = parse_response(text)
        assert result["type"] == "tool_call"
        assert result["tool"] == "calculator"
        assert result["input"] == "2 + 2"

    def test_parses_final_answer_with_thinking(self):
        text = "<think>Let me answer</think>\nFINAL_ANSWER: Hello there!"
        result = parse_response(text)
        assert result["type"] == "final_answer"
        assert "Hello" in result["content"]

    def test_unknown_format(self):
        result = parse_response("Just some regular text without format")
        assert result["type"] == "unknown"
        assert "regular text" in result["content"]

    def test_tool_call_with_spaces(self):
        result = parse_response("TOOL_CALL:  web_search  | INPUT:  current bitcoin price  ")
        assert result["type"] == "tool_call"
        assert result["tool"] == "web_search"
        assert "bitcoin" in result["input"]

    def test_multiline_final_answer(self):
        text = "FINAL_ANSWER: Line one\nLine two\nLine three"
        result = parse_response(text)
        assert result["type"] == "final_answer"
        assert "Line one" in result["content"]

    def test_tool_call_various_tools(self):
        tools = ["calculator", "datetime", "web_search", "wikipedia", "memory"]
        for tool in tools:
            result = parse_response(f"TOOL_CALL: {tool} | INPUT: test input")
            assert result["type"] == "tool_call"
            assert result["tool"] == tool
