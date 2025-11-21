"""
Tests for analysis module
"""

import pytest
from src.analysis.text_processor import TextProcessor

def test_clean_text():
    processor = TextProcessor()
    text = "Check this out @user https://example.com"
    cleaned = processor.clean_text(text)
    assert "@user" not in cleaned
    assert "https://" not in cleaned
