"""
Memory Tool — Updated with Fuzzy Key Matching

Problem solved: The LLM might save as "favorite color" but recall as
"favorite_color" or "fav color". Without fuzzy matching, the recall
would fail even though the data exists.

Solution: On recall, if an exact match isn't found, we normalize both
the query and stored keys (strip spaces, underscores, common words)
and try again.

Interview talking point: "I discovered during testing that the LLM
would format memory keys inconsistently — saving as 'favorite color'
but recalling as 'favorite_color'. I fixed it by adding key
normalization and fuzzy matching on recall."
"""

import json
import os
from agent.tools.base import BaseTool


MEMORY_DIR = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "user_memory.json")

# Words to strip during normalization — these add no meaning to keys
NOISE_WORDS = {"my", "the", "a", "an", "is", "are", "was", "that", "this"}


def _normalize_key(key: str) -> str:
    """
    Normalize a key for fuzzy matching.

    Steps:
    1. Lowercase
    2. Replace underscores and hyphens with spaces
    3. Remove noise words (my, the, a, etc.)
    4. Collapse multiple spaces
    5. Strip whitespace

    "my_favorite_color" → "favorite color"
    "favorite color"    → "favorite color"
    "My Favorite Color" → "favorite color"
    """
    key = key.lower()
    key = key.replace("_", " ").replace("-", " ")
    words = [w for w in key.split() if w not in NOISE_WORDS]
    return " ".join(words).strip()


class MemoryTool(BaseTool):
    """
    Tool for saving and recalling information across sessions.

    Usage by the LLM:
        TOOL_CALL: memory | INPUT: save name = Abhishek
        TOOL_CALL: memory | INPUT: recall name
        TOOL_CALL: memory | INPUT: recall all
        TOOL_CALL: memory | INPUT: forget name
    """

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return (
            "Use this tool to save or recall information that should persist "
            "across conversations. "
            "To save: 'save key = value' (e.g., 'save name = Abhishek'). "
            "To recall: 'recall key' (e.g., 'recall name'). "
            "To see everything saved: 'recall all'. "
            "Use this when the user asks you to remember something, or when "
            "they ask about something they previously told you to remember."
        )

    def run(self, tool_input: str) -> str:
        cleaned = tool_input.strip()

        try:
            lower = cleaned.lower()

            if lower.startswith("save"):
                return self._save(cleaned[4:].strip())
            elif lower.startswith("recall"):
                return self._recall(cleaned[6:].strip())
            elif lower.startswith("delete") or lower.startswith("forget"):
                key_part = cleaned.split(maxsplit=1)
                if len(key_part) > 1:
                    return self._delete(key_part[1].strip())
                return "Error: Please specify what to forget (e.g., 'forget email')."
            else:
                return (
                    "Error: Unknown memory operation. "
                    "Use 'save key = value', 'recall key', 'recall all', "
                    "or 'forget key'."
                )

        except Exception as e:
            return f"Error with memory operation: {str(e)}"

    def _save(self, input_str: str) -> str:
        """Parse and save a key-value pair."""
        key, value = None, None

        for delimiter in ["=", ":", " is "]:
            if delimiter in input_str:
                parts = input_str.split(delimiter, 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    break

        if not key or not value:
            return (
                "Error: Could not parse save command. "
                "Use format: 'save key = value' (e.g., 'save name = Abhishek')."
            )

        memory = self._load_memory()
        memory[key] = value
        self._write_memory(memory)

        return f"Saved: {key} = {value}"

    def _recall(self, input_str: str) -> str:
        """
        Recall a value by key with fuzzy matching.

        Match priority:
        1. Exact match (case-insensitive)
        2. Normalized match (strip noise words, underscores)
        """
        memory = self._load_memory()

        if not memory:
            return "No memories saved yet."

        key = input_str.strip().lower()

        if key in ("all", "everything", ""):
            lines = ["Here's everything I remember:"]
            for k, v in memory.items():
                lines.append(f"  - {k}: {v}")
            return "\n".join(lines)

        # Priority 1: Exact match
        if key in memory:
            return f"{key}: {memory[key]}"

        # Priority 2: Normalized match
        normalized_query = _normalize_key(key)
        for stored_key, value in memory.items():
            if _normalize_key(stored_key) == normalized_query:
                return f"{stored_key}: {value}"

        available = ", ".join(memory.keys())
        return (
            f"No memory found for '{key}'. "
            f"Available memories: {available}"
        )

    def _delete(self, input_str: str) -> str:
        """Delete a memory by key with fuzzy matching."""
        memory = self._load_memory()
        key = input_str.strip().lower()

        # Exact match
        if key in memory:
            del memory[key]
            self._write_memory(memory)
            return f"Forgotten: {key}"

        # Normalized match
        normalized_query = _normalize_key(key)
        for stored_key in list(memory.keys()):
            if _normalize_key(stored_key) == normalized_query:
                del memory[stored_key]
                self._write_memory(memory)
                return f"Forgotten: {stored_key}"

        return f"No memory found for '{key}', nothing to forget."

    def _load_memory(self) -> dict:
        if not os.path.exists(MEMORY_FILE):
            return {}

        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _write_memory(self, memory: dict):
        os.makedirs(MEMORY_DIR, exist_ok=True)

        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
