# Twitter/Nitter Data Scraper & Analysis Project

A comprehensive Python-based project for scraping, analyzing, and visualizing Twitter data using Nitter instances.

## 🚀 Features

- **Efficient Web Scraping**: Optimized Selenium-based scraper for Nitter
- **Data Collection**: Comprehensive tweet metadata extraction
- **Sentiment Analysis**: Text processing and sentiment scoring
- **Data Visualization**: Interactive plots and charts
- **Checkpoint System**: Resume interrupted scraping sessions
- **Flexible Export**: Multiple CSV export modes

## 📋 Project Structure

```
twitter-scraper-project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py
│
├── src/
│   ├── scraper/          # Web scraping modules
│   ├── analysis/         # Data analysis and sentiment
│   ├── visualization/    # Plotting and charts
│   └── utils/           # Helper functions
│
├── notebooks/           # Jupyter notebooks for exploration
├── tests/              # Unit tests
├── config/             # Configuration files
├── data/               # Data storage (gitignored)
│   ├── raw/
│   ├── processed/
│   └── results/
└── docs/               # Documentation
```

## 🔧 Installation

1. Navigate to project directory:
```bash
cd twitter-scraper-project
```

2. Create a virtual environment:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Basic Scraping

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

### Command Line

```bash
python src/scraper/nitter_scraper.py
```

### Tweet Types

- `original`: Only original tweets (no retweets or quotes)
- `original_and_quotes`: Original tweets + quote tweets
- `all`: Everything including retweets

### CSV Save Modes

- `timestamp`: Add timestamp to filename (recommended)
- `increment`: Add number suffix (_1, _2, etc.)
- `merge`: Merge with existing CSV, avoid duplicates
- `ask`: Prompt before overwriting

## 📊 Data Dictionary

Each scraped tweet contains:

| Field | Type | Description |
|-------|------|-------------|
| tweet_id | string | Unique tweet identifier |
| text | string | Tweet content |
| created_at | datetime | Publication timestamp |
| lang | string | Detected language code |
| user_id_hashed | string | Anonymized user ID |
| retweet_count | integer | Number of retweets |
| like_count | integer | Number of likes |
| comment_count | integer | Number of replies |
| is_reply | boolean | Is this a reply? |
| reply_to_id | string | Parent tweet ID (if reply) |
| is_retweet | boolean | Is this a retweet? |
| is_quote | boolean | Is this a quote tweet? |
| urls | json | Extracted URLs |
| hashtags | json | Extracted hashtags |
| mentions | json | Extracted mentions |
| media | json | Media attachments info |

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/
```

With coverage:

```bash
pytest --cov=src tests/
```

## ⚠️ Important Notes

- Respect rate limits and terms of service
- Data is collected for research/educational purposes
- User IDs are hashed for privacy
- Nitter instances may have varying availability

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Selenium WebDriver
- Nitter project
- All contributors

---

**Disclaimer**: This tool is for educational and research purposes. Always comply with applicable laws and platform terms of service.