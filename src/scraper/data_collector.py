"""
Data Collector Module
Handles data collection orchestration
"""

from .nitter_scraper import scrape_x_posts, export_to_csv

def collect_user_data(username, num_scrolls=50, tweet_type="original"):
    """
    Collect data for a specific user
    
    Args:
        username: Twitter username to scrape
        num_scrolls: Number of scrolls to perform
        tweet_type: Type of tweets to collect
        
    Returns:
        List of tweet dictionaries
    """
    tweets = scrape_x_posts(username, num_scrolls, tweet_type)
    return tweets
