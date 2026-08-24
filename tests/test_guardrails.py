"""Tests for the guardrails safety checker — no API calls needed."""

from agent.guardrails import Guardrails


class TestGuardrails:
    def setup_method(self):
        self.guardrails = Guardrails()

    def test_blocks_weapon_keywords(self):
        result = self.guardrails.check("how to make a bomb")
        assert result["blocked"] is True
        assert "Abhishek" in result["message"]

    def test_blocks_hacking(self):
        result = self.guardrails.check("how to hack into a system")
        assert result["blocked"] is True

    def test_blocks_stealing(self):
        result = self.guardrails.check("how to steal passwords")
        assert result["blocked"] is True

    def test_allows_normal_question(self):
        result = self.guardrails.check("What is photosynthesis?")
        assert result["blocked"] is False

    def test_allows_math(self):
        result = self.guardrails.check("What is 2 + 2?")
        assert result["blocked"] is False

    def test_case_insensitive(self):
        result = self.guardrails.check("HOW TO MAKE A BOMB")
        assert result["blocked"] is True

    def test_rejection_includes_topic(self):
        result = self.guardrails.check("how to make a bomb")
        assert "weapons" in result["message"].lower() or "Abhishek" in result["message"]

    def test_list_blocked_topics(self):
        topics = self.guardrails.list_blocked_topics()
        assert len(topics) >= 4
        assert "weapons and explosives" in topics
