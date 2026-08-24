"""Tests for the datetime tool — no API calls needed."""

from agent.tools.datetime_tool import DateTimeTool


class TestDateTimeTool:
    def setup_method(self):
        self.tool = DateTimeTool()

    def test_name(self):
        assert self.tool.name == "datetime"

    def test_description_exists(self):
        assert len(self.tool.description) > 0

    def test_current_date(self):
        result = self.tool.run("current date")
        assert "UTC" in result
        assert "Current" in result

    def test_current_time(self):
        result = self.tool.run("current time")
        assert "UTC" in result

    def test_day_of_week(self):
        result = self.tool.run("day of week")
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(day in result for day in days)

    def test_current_datetime(self):
        result = self.tool.run("current datetime")
        assert "UTC" in result
        assert "at" in result

    def test_unknown_input_returns_everything(self):
        """Unknown input should return full datetime rather than error."""
        result = self.tool.run("something random")
        assert "UTC" in result

    def test_returns_string(self):
        result = self.tool.run("current date")
        assert isinstance(result, str)
