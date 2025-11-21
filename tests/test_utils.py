"""
Tests for utility functions
"""

import pytest
from src.utils.helpers import get_timestamp

def test_get_timestamp():
    timestamp = get_timestamp()
    assert len(timestamp) == 15  # YYYYMMDD_HHMMSS format
