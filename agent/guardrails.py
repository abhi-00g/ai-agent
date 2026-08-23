"""
Safety Guardrails

This module checks user messages BEFORE they reach the agent loop.
If a message matches a blocked topic, it returns a rejection message
immediately — the LLM is never called.

Why check before the LLM, not after?
1. Saves API credits — blocked questions never hit Gemini
2. Prevents the LLM from generating harmful content that we'd then
   have to filter out (the content would still be in memory)
3. Faster response — no network round-trip to the LLM

Why YAML instead of hardcoded Python?
  - Non-developers can edit the guardrails without touching code
  - You can add new blocked topics without redeploying
  - In an interview: "I used configuration-driven guardrails so the
    safety rules are decoupled from the application logic. Adding a
    new blocked topic is a YAML edit, not a code change."

The current approach uses keyword matching, which is simple but effective
for obvious cases. In a production system, you'd add LLM-based classification
for subtler attempts (e.g., rephrasing blocked questions to avoid keywords).
You can mention this as a scaling consideration in interviews.
"""

import os
import yaml


# Path to the guardrails config file
GUARDRAILS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "guardrails.yaml"
)


class Guardrails:
    """
    Loads safety guardrails from YAML and checks messages against them.

    Usage:
        guardrails = Guardrails()
        result = guardrails.check("how to make a bomb")
        if result["blocked"]:
            print(result["message"])  # Returns rejection message
    """

    def __init__(self):
        self.blocked_topics = []
        self.rejection_template = ""
        self._load_config()

    def _load_config(self):
        """
        Load the guardrails YAML config file.

        If the file doesn't exist or is malformed, guardrails are disabled
        (empty blocked topics list). This is a deliberate choice — we don't
        want a missing config file to crash the agent. We log a warning
        instead so the developer knows guardrails are inactive.
        """
        if not os.path.exists(GUARDRAILS_FILE):
            print(
                "Warning: guardrails.yaml not found. "
                "Safety guardrails are disabled."
            )
            return

        try:
            with open(GUARDRAILS_FILE, "r") as f:
                config = yaml.safe_load(f)

            if not config:
                return

            self.rejection_template = config.get("rejection_message", "").strip()
            self.blocked_topics = config.get("blocked_topics", [])

        except (yaml.YAMLError, IOError) as e:
            print(f"Warning: Could not load guardrails.yaml: {e}")
            print("Safety guardrails are disabled.")

    def check(self, message: str) -> dict:
        """
        Check a user message against blocked topics.

        Args:
            message: The user's raw input string.

        Returns:
            A dict with:
            - {"blocked": False} if the message is safe
            - {"blocked": True, "message": "rejection text", "topic": "matched topic"}
              if the message matches a blocked topic

        The check is case-insensitive. We normalize both the message and
        keywords to lowercase before comparing.
        """
        if not self.blocked_topics:
            return {"blocked": False}

        message_lower = message.lower()

        for topic in self.blocked_topics:
            topic_name = topic.get("name", "restricted content")
            keywords = topic.get("keywords", [])

            for keyword in keywords:
                if keyword.lower() in message_lower:
                    # Match found — build rejection message
                    rejection = self.rejection_template.replace(
                        "{topic}", topic_name
                    )

                    # Fallback if no template was configured
                    if not rejection:
                        rejection = (
                            f"I'm unable to help with questions about "
                            f"{topic_name}."
                        )

                    return {
                        "blocked": True,
                        "message": rejection,
                        "topic": topic_name,
                    }

        return {"blocked": False}

    def list_blocked_topics(self) -> list[str]:
        """Return list of blocked topic names (for debugging)."""
        return [t.get("name", "unknown") for t in self.blocked_topics]
