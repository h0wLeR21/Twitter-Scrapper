# Installation Guide

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Chrome browser (for Selenium)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd twitter-scraper-project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "from src.scraper.nitter_scraper import scrape_x_posts; print('Installation successful!')"
```

## Troubleshooting

### ChromeDriver Issues
- ChromeDriver is automatically managed by webdriver-manager
- If issues persist, manually download from: https://chromedriver.chromium.org/

### Permission Errors
- Run terminal as administrator (Windows)
- Use `sudo` for installation (Mac/Linux)

### Module Not Found
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
