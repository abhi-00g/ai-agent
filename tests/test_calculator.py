"""Tests for the calculator tool — no API calls needed."""

from agent.tools.calculator import CalculatorTool, safe_eval


class TestSafeEval:
    def test_addition(self):
        assert safe_eval("2 + 3") == 5

    def test_multiplication(self):
        assert safe_eval("245 * 18") == 4410

    def test_division(self):
        assert safe_eval("1000 / 8") == 125

    def test_power(self):
        assert safe_eval("2 ** 10") == 1024

    def test_complex_expression(self):
        assert safe_eval("(100 + 50) * 3") == 450

    def test_floor_division(self):
        assert safe_eval("17 // 3") == 5

    def test_modulo(self):
        assert safe_eval("17 % 3") == 2

    def test_negative_numbers(self):
        assert safe_eval("-5 + 3") == -2

    def test_rejects_function_calls(self):
        """Should reject anything that isn't pure math."""
        tool = CalculatorTool()
        result = tool.run("__import__('os').system('ls')")
        assert "Error" in result

    def test_division_by_zero(self):
        tool = CalculatorTool()
        result = tool.run("10 / 0")
        assert "Error" in result


class TestCalculatorTool:
    def setup_method(self):
        self.tool = CalculatorTool()

    def test_name(self):
        assert self.tool.name == "calculator"

    def test_description_exists(self):
        assert len(self.tool.description) > 0

    def test_run_simple(self):
        result = self.tool.run("2 + 2")
        assert "4" in result

    def test_run_returns_string(self):
        result = self.tool.run("100 * 5")
        assert isinstance(result, str)

    def test_whole_number_no_decimal(self):
        """Whole numbers should show as 4410, not 4410.0"""
        result = self.tool.run("245 * 18")
        assert "4410" in result
        assert "4410.0" not in result
