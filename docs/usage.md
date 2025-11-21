# Usage Guide

## Basic Usage

### Scraping Tweets

```python
from src.scraper.nitter_scraper import scrape_x_posts, export_to_csv

# Scrape tweets
tweets = scrape_x_posts(
    username="example_user",
    num_scrolls=50,
    tweet_type="original"
)

# Export to CSV
export_to_csv(tweets, "example_user", mode="timestamp")
```

### Tweet Types

- **original**: Only original tweets
- **original_and_quotes**: Original + quote tweets
- **all**: Everything including retweets

### CSV Save Modes

- **timestamp**: Adds timestamp to filename
- **increment**: Adds number suffix
- **merge**: Merges with existing CSV
- **ask**: Prompts before overwriting

## Advanced Usage

### Custom Configuration

```python
import yaml
from src.utils.helpers import load_config

config = load_config('config/config.yaml')
num_scrolls = config['scraping']['default_scrolls']
```

### Sentiment Analysis

```python
from src.analysis.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
tweets_with_sentiment = analyzer.analyze_tweets(tweets)
```

### Visualization

```python
from src.visualization.plot_generator import PlotGenerator

plotter = PlotGenerator()
plotter.plot_tweet_timeline(tweets)
plotter.plot_engagement_stats(tweets)
```

## Command Line Usage

```bash
python src/scraper/nitter_scraper.py
```

Follow the interactive prompts.
