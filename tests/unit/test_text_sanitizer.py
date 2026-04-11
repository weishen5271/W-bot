"""Unit tests for text_sanitizer module."""

from __future__ import annotations

from w_bot.agents.core.text_sanitizer import sanitize_user_text


class TestSanitizeUserText:
    """Tests for sanitize_user_text function."""

    def test_normal_text_unchanged(self) -> None:
        """Test normal text without control characters is unchanged."""
        text = "Hello, world! This is a normal text."
        result = sanitize_user_text(text)
        assert result == text

    def test_crlf_converted_to_lf(self) -> None:
        """Test CRLF line endings are converted to LF."""
        text = "line1\r\nline2\r\nline3"
        result = sanitize_user_text(text)
        assert "\r\n" not in result
        assert result == "line1\nline2\nline3"

    def test_cr_converted_to_lf(self) -> None:
        """Test standalone CR is converted to LF."""
        text = "line1\rline2\rline3"
        result = sanitize_user_text(text)
        assert "\r" not in result
        assert result == "line1\nline2\nline3"

    def test_nbsp_converted_to_space(self) -> None:
        """Test non-breaking space is converted to regular space."""
        text = "hello\u00a0world"
        result = sanitize_user_text(text)
        assert "\u00a0" not in result
        assert " " in result

    def test_zero_width_space_removed(self) -> None:
        """Test zero-width space is removed."""
        text = "hello\u200bworld"
        result = sanitize_user_text(text)
        assert "\u200b" not in result
        assert result == "helloworld"

    def test_control_characters_removed(self) -> None:
        """Test control characters (Cc category) are removed."""
        # Bell character (control)
        text = "hello\x07world"
        result = sanitize_user_text(text)
        assert "\x07" not in result

        # Null character (control)
        text = "hello\x00world"
        result = sanitize_user_text(text)
        assert "\x00" not in result

    def test_newline_and_tab_preserved(self) -> None:
        """Test newline and tab characters are preserved."""
        text = "line1\nline2\twith\ttab"
        result = sanitize_user_text(text)
        assert "\n" in result
        assert "\t" in result

    def test_empty_string(self) -> None:
        """Test empty string returns empty string."""
        result = sanitize_user_text("")
        assert result == ""

    def test_none_input(self) -> None:
        """Test None input returns empty string."""
        result = sanitize_user_text(None)
        assert result == ""

    def test_chinese_text(self) -> None:
        """Test Chinese text is handled correctly."""
        text = "你好，\n世界！\t你好"
        result = sanitize_user_text(text)
        assert "你好" in result
        assert "\n" in result
        assert "\t" in result

    def test_mixed_content(self) -> None:
        """Test mixed content with various special characters."""
        text = "Hello\u00a0World\r\nLine2\u200b\rLine3\x00end"
        result = sanitize_user_text(text)
        assert "Hello World" in result or ("Hello" in result and "World" in result)
        assert "\n" in result
        assert "\r" not in result
        assert "\u200b" not in result
        assert "\x00" not in result
