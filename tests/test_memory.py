"""Tests for the memory tool — no API calls needed."""

import os
import json
import pytest
from agent.tools.memory import MemoryTool, MEMORY_FILE, MEMORY_DIR


class TestMemoryTool:
    def setup_method(self):
        """Clean up memory file before each test."""
        self.tool = MemoryTool()
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)

    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        if os.path.exists(MEMORY_DIR) and not os.listdir(MEMORY_DIR):
            os.rmdir(MEMORY_DIR)

    def test_name(self):
        assert self.tool.name == "memory"

    def test_description_exists(self):
        assert len(self.tool.description) > 0

    def test_save_with_equals(self):
        result = self.tool.run("save name = Abhishek")
        assert "Saved" in result
        assert "name" in result

    def test_save_with_colon(self):
        result = self.tool.run("save email: test@test.com")
        assert "Saved" in result

    def test_save_with_is(self):
        result = self.tool.run("save favorite_color is blue")
        assert "Saved" in result

    def test_recall_existing(self):
        self.tool.run("save name = Abhishek")
        result = self.tool.run("recall name")
        assert "Abhishek" in result

    def test_recall_nonexistent(self):
        result = self.tool.run("recall something")
        assert "No memor" in result

    def test_recall_all_empty(self):
        result = self.tool.run("recall all")
        assert "No memories" in result

    def test_recall_all_with_data(self):
        self.tool.run("save name = Abhishek")
        self.tool.run("save age = 24")
        result = self.tool.run("recall all")
        assert "Abhishek" in result
        assert "24" in result

    def test_delete(self):
        self.tool.run("save name = Abhishek")
        result = self.tool.run("delete name")
        assert "Forgotten" in result

        recall = self.tool.run("recall name")
        assert "No memor" in recall

    def test_persistence(self):
        """Data should survive creating a new tool instance."""
        self.tool.run("save city = Boston")
        new_tool = MemoryTool()
        result = new_tool.run("recall city")
        assert "Boston" in result

    def test_case_insensitive_keys(self):
        self.tool.run("save Name = Abhishek")
        result = self.tool.run("recall name")
        assert "Abhishek" in result

    def test_invalid_operation(self):
        result = self.tool.run("update name = Abhishek")
        assert "Error" in result
