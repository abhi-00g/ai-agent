"""
DateTime Tool

Provides current date and time information. The LLM uses this when a user
asks anything time-related: "what day is it?", "how many days until Christmas?",
"what time is it in Tokyo?".

Why is this a separate tool instead of just letting the LLM answer?
Because LLMs don't know the current date. Their training data has a cutoff,
and they generate text based on patterns, not real-time information. If you
ask Gemini "what's today's date?" without this tool, it will either refuse
or hallucinate. This tool gives it access to actual system time.
"""

from datetime import datetime, timezone
from agent.tools.base import BaseTool


class DateTimeTool(BaseTool):
    """
    Tool for getting current date/time information.

    Usage by the LLM:
        TOOL_CALL: datetime | INPUT: current date
        TOOL_CALL: datetime | INPUT: current time
        TOOL_CALL: datetime | INPUT: current datetime
        TOOL_CALL: datetime | INPUT: day of week
    """

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return (
            "Use this tool to get the current date, time, or day of the week. "
            "Send one of: 'current date', 'current time', 'current datetime', "
            "or 'day of week'. Always use this tool instead of guessing the "
            "current date or time."
        )

    def run(self, tool_input: str) -> str:
        try:
            now = datetime.now(timezone.utc)
            query = tool_input.strip().lower()

            if "time" in query and "date" in query:
                # "current datetime" or "date and time"
                return now.strftime(
                    "Current UTC date and time: %A, %B %d, %Y at %I:%M %p UTC"
                )
            elif "date" in query:
                return now.strftime("Current UTC date: %A, %B %d, %Y")
            elif "time" in query:
                return now.strftime("Current UTC time: %I:%M %p UTC")
            elif "day" in query:
                return now.strftime("Current day of the week: %A")
            else:
                # If the input doesn't match expected patterns, return
                # everything — better to give too much info than too little
                return now.strftime(
                    "Current UTC date and time: %A, %B %d, %Y at %I:%M %p UTC"
                )

        except Exception as e:
            return f"Error getting date/time: {str(e)}"
