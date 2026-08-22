"""
Base class for all tools.

This is the Strategy pattern in action. Every tool — calculator, web search,
memory, whatever — must inherit from this class and implement the same
interface: name, description, and run().

Why does this matter?
- The agent loop doesn't know or care what a tool does. It just calls
  tool.run(input) and gets a string back. This means you can add 50 new
  tools without changing a single line in the agent loop.
- The registry auto-generates the system prompt from tool descriptions,
  so the LLM knows what each tool does without you manually editing prompts.
- In interviews, you can say: "I used the Strategy pattern so tools are
  pluggable — adding a new tool is just creating a new class."
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Every tool must define:
      - name: a short identifier (e.g., "calculator"). This is what the LLM
        writes in its TOOL_CALL response to select this tool.
      - description: a sentence explaining what this tool does and when to use
        it. This goes directly into the system prompt for the LLM.
      - run(tool_input): takes a string input, does the work, returns a string
        result. The agent loop calls this when the LLM selects this tool.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique identifier for this tool (e.g., 'calculator')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        One-line description of what this tool does.
        This is shown to the LLM in the system prompt, so write it as an
        instruction: "Use this tool when you need to..."
        """
        pass

    @abstractmethod
    def run(self, tool_input: str) -> str:
        """
        Execute the tool with the given input and return the result as a string.

        Args:
            tool_input: The input string from the LLM's tool call.

        Returns:
            A string containing the result. Even if the result is a number,
            return it as a string — the LLM reads text, not Python objects.
        """
        pass
