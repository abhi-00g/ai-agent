"""
Memory Tool — Updated with Public Demo Mode

Changes from previous version:
- PUBLIC_DEMO mode: when enabled, memory writes (save/delete) are disabled.
  Memory is pre-loaded with curated professional facts and read-only.
- Fuzzy key matching is unchanged.
- For local development, set PUBLIC_DEMO=false in .env to enable full access.
"""

import json
import os
from agent.tools.base import BaseTool
from agent.config import PUBLIC_DEMO


MEMORY_DIR = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "user_memory.json")

# Words to strip during normalization
NOISE_WORDS = {"my", "the", "a", "an", "is", "are", "was", "that", "this"}

# Pre-loaded professional facts for public demo mode
DEMO_MEMORY = {
    "creator": "Venkata Krishna Raj Abhishek Gade (Abhishek)",
    "creator location": "Boston, Massachusetts",
    "creator education": "MS Software Engineering Systems at Northeastern University (GPA 3.81), BTech CSE from GRIET Hyderabad",
    "creator skills": "Python, TypeScript, React, FastAPI, Node.js, PostgreSQL, Redis, AWS, Terraform, Docker, FAISS, LangChain",
    "creator portfolio": "https://portfolio-taupe-seven-64piyh5mxj.vercel.app/",
    "creator github": "https://github.com/abhi-00g",
    "creator linkedin": "https://www.linkedin.com/in/venkata-krishna-raj-abhishek-gade-717147230/",
    "atlas tagline": "I carry the weight so you don't have to.",
    "atlas tools": "calculator, datetime, web_search, wikipedia, memory",
}

PUBLIC_WRITE_DECLINE = (
    "Memory writes are disabled on the public demo to keep things clean. "
    "In local development, ATLAS has full persistent memory with fuzzy key matching. "
    "You can still ask me to recall anything I already know!"
)


def _normalize_key(key: str) -> str:
    """Normalize a key for fuzzy matching."""
    key = key.lower()
    key = key.replace("_", " ").replace("-", " ")
    words = [w for w in key.split() if w not in NOISE_WORDS]
    return " ".join(words).strip()


class MemoryTool(BaseTool):
    """
    Tool for saving and recalling information across sessions.

    In PUBLIC_DEMO mode, writes are disabled and memory is pre-loaded
    with curated facts. In local mode, full read/write access.
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
                if PUBLIC_DEMO:
                    return PUBLIC_WRITE_DECLINE
                return self._save(cleaned[4:].strip())

            elif lower.startswith("recall"):
                return self._recall(cleaned[6:].strip())

            elif lower.startswith("delete") or lower.startswith("forget"):
                if PUBLIC_DEMO:
                    return PUBLIC_WRITE_DECLINE
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
        """Recall a value by key with fuzzy matching."""
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

        if key in memory:
            del memory[key]
            self._write_memory(memory)
            return f"Forgotten: {key}"

        normalized_query = _normalize_key(key)
        for stored_key in list(memory.keys()):
            if _normalize_key(stored_key) == normalized_query:
                del memory[stored_key]
                self._write_memory(memory)
                return f"Forgotten: {stored_key}"

        return f"No memory found for '{key}', nothing to forget."

    def _load_memory(self) -> dict:
        """
        Load memory. In PUBLIC_DEMO mode, returns the curated demo memory.
        In local mode, reads from the JSON file.
        """
        if PUBLIC_DEMO:
            return dict(DEMO_MEMORY)

        if not os.path.exists(MEMORY_FILE):
            return {}

        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _write_memory(self, memory: dict):
        """Write memory to file. Only called in local mode (PUBLIC_DEMO=false)."""
        os.makedirs(MEMORY_DIR, exist_ok=True)

        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
