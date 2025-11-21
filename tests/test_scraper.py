"""
Tests for scraper module
"""

import pytest
from src.scraper.nitter_scraper import hash_username, extract_tweet_id

def test_hash_username():
    username = "testuser"
    hashed = hash_username(username)
    assert len(hashed) == 16
    assert hash_username(username) == hashed  # Consistent

def test_extract_tweet_id():
    url = "https://nitter.net/user/status/1234567890"
    tweet_id = extract_tweet_id(url)
    assert tweet_id == "1234567890"
