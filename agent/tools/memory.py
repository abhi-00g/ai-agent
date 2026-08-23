"""
Memory Tool

Provides persistent memory across sessions using a JSON file.
The agent can save facts (like the user's name or email) and recall
them later — even after the Python process restarts.

Two operations:
  - save: stores a key-value pair (e.g., save name = Abhishek)
  - recall: retrieves a value by key (e.g., recall name)

There's also a third implicit operation: "recall all" or "what do you
remember" which dumps everything stored.

Why JSON instead of a database?
  - This project already demonstrates PostgreSQL expertise via the
    AI Cost Dashboard (SQLAlchemy, Alembic, asyncpg).
  - For a single-user agent, JSON is the right tool for the job.
  - In an interview: "I chose JSON because the access pattern is simple
    key-value reads and writes for a single user. If I were scaling to
    multiple users, I'd move to PostgreSQL with per-user partitioning."

Design decisions:
  - The file is created on first save, not on startup. This avoids
    creating empty files that clutter the project.
  - Every save immediately writes to disk (not batched). This ensures
    data survives even if the process crashes mid-conversation.
  - Keys are case-insensitive (stored lowercase) to prevent the user
    from accidentally saving "Name" and "name" as separate entries.
  - The tool parses natural language inputs like "save my name is Abhishek"
    into key-value pairs. The LLM sends free-form text, not structured JSON.
"""

import json
import os
from agent.tools.base import BaseTool


# Where the memory file lives. The memory/ directory is gitignored
# so personal data never gets committed to the repo.
MEMORY_DIR = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "user_memory.json")


class MemoryTool(BaseTool):
    """
    Tool for saving and recalling information across sessions.

    Usage by the LLM:
        TOOL_CALL: memory | INPUT: save name = Abhishek
        TOOL_CALL: memory | INPUT: save email = gade.venk@northeastern.edu
        TOOL_CALL: memory | INPUT: recall name
        TOOL_CALL: memory | INPUT: recall all
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
            # Determine the operation: save or recall
            lower = cleaned.lower()

            if lower.startswith("save"):
                return self._save(cleaned[4:].strip())
            elif lower.startswith("recall"):
                return self._recall(cleaned[6:].strip())
            elif lower.startswith("delete") or lower.startswith("forget"):
                # Bonus: let the agent delete memories too
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
        """
        Parse and save a key-value pair.

        Accepts formats like:
          - "name = Abhishek"
          - "name: Abhishek"
          - "name is Abhishek"

        The flexibility matters because the LLM won't always use the exact
        same format. By supporting multiple delimiters, we reduce parse failures.
        """
        # Try splitting on common delimiters
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

        # Load existing memory, add new entry, write back
        memory = self._load_memory()
        memory[key] = value
        self._write_memory(memory)

        return f"Saved: {key} = {value}"

    def _recall(self, input_str: str) -> str:
        """
        Recall a value by key, or dump all memories.
        """
        memory = self._load_memory()

        if not memory:
            return "No memories saved yet."

        key = input_str.strip().lower()

        # "recall all" or "recall everything" dumps all memories
        if key in ("all", "everything", ""):
            lines = ["Here's everything I remember:"]
            for k, v in memory.items():
                lines.append(f"  - {k}: {v}")
            return "\n".join(lines)

        # Look up specific key
        if key in memory:
            return f"{key}: {memory[key]}"

        # Key not found — tell the LLM what IS available
        available = ", ".join(memory.keys())
        return (
            f"No memory found for '{key}'. "
            f"Available memories: {available}"
        )

    def _delete(self, input_str: str) -> str:
        """Delete a memory by key."""
        memory = self._load_memory()
        key = input_str.strip().lower()

        if key in memory:
            del memory[key]
            self._write_memory(memory)
            return f"Forgotten: {key}"

        return f"No memory found for '{key}', nothing to forget."

    def _load_memory(self) -> dict:
        """
        Load the memory file from disk. Returns empty dict if file
        doesn't exist yet (first-time use).
        """
        if not os.path.exists(MEMORY_FILE):
            return {}

        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If the file is corrupted, start fresh rather than crash
            return {}

    def _write_memory(self, memory: dict):
        """
        Write memory dict to disk. Creates the memory/ directory if
        it doesn't exist.

        We write immediately on every save (not batched) because:
        1. The user expects "remember X" to actually persist
        2. If the process crashes, we don't lose unsaved data
        3. The file is tiny — writing is effectively instant
        """
        os.makedirs(MEMORY_DIR, exist_ok=True)

        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
