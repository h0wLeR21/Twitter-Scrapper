#!/usr/bin/env python3
"""
Quick Start Script for Twitter-Scrapper
Run this to test your installation
"""

print("Testing Twitter Scraper Project Installation...")
print("=" * 60)

# Test imports
try:
    from src.scraper.nitter_scraper import scrape_x_posts
    print("✓ Scraper module loaded successfully")
except ImportError as e:
    print(f"✗ Could not import scraper: {e}")

try:
    from src.analysis.sentiment_analyzer import SentimentAnalyzer
    print("✓ Analysis module loaded successfully")
except ImportError as e:
    print(f"✗ Could not import analyzer: {e}")

try:
    from src.visualization.plot_generator import PlotGenerator
    print("✓ Visualization module loaded successfully")
except ImportError as e:
    print(f"✗ Could not import plotter: {e}")

try:
    from src.utils.helpers import load_config
    print("✓ Utilities module loaded successfully")
except ImportError as e:
    print(f"✗ Could not import helpers: {e}")

print("=" * 60)
print("✓ Installation test complete!")
print("\nYou're ready to start scraping!")
