# API Documentation

## Core Functions

### scrape_x_posts()

```python
def scrape_x_posts(username, num_scrolls=10, tweet_type="original", checkpoint_file=None)
```

Scrape tweets from a Nitter user profile.

**Parameters:**
- `username` (str): Twitter username to scrape
- `num_scrolls` (int): Number of page scrolls (default: 10)
- `tweet_type` (str): Type filter - "original", "original_and_quotes", "all"
- `checkpoint_file` (str): Path to checkpoint file (optional)

**Returns:**
- List of tweet dictionaries

**Example:**
```python
tweets = scrape_x_posts("elonmusk", num_scrolls=50, tweet_type="original")
```

---

### export_to_csv()

```python
def export_to_csv(tweets, username, mode="timestamp")
```

Export tweets to CSV file.

**Parameters:**
- `tweets` (list): List of tweet dictionaries
- `username` (str): Username for filename
- `mode` (str): Save mode - "timestamp", "increment", "merge", "ask"

**Returns:**
- Filename (str) of saved CSV

**Example:**
```python
filename = export_to_csv(tweets, "elonmusk", mode="merge")
```

---

## Utility Functions

### hash_username()

```python
def hash_username(username)
```

Create consistent hash for username (cached).

---

### detect_language()

```python
def detect_language(text)
```

Detect language of text with caching.

---

### extract_tweet_id()

```python
def extract_tweet_id(url)
```

Extract tweet ID from URL (cached).

---

## Analysis Functions

### SentimentAnalyzer

```python
from src.analysis.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
scores = analyzer.analyze_tweet("Great day!")
```

**Returns:**
```python
{
    'neg': 0.0,
    'neu': 0.408,
    'pos': 0.592,
    'compound': 0.6249
}
```
