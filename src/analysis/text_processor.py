import pandas as pd
import numpy as np
import re
import os
import json
from datetime import datetime
import unicodedata

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    print("Note: 'emoji' package not installed. Emoji removal will use fallback method.")

# ---------------------------
# Configuration
# ---------------------------

RAW_DATA_PATH = r"D:\code\Edure\New folder\Twitter-Scrapper\data\raw"
PROCESSED_DATA_PATH = r"D:\code\Edure\New folder\Twitter-Scrapper\data\processed"

# ---------------------------
# Text Cleaning Functions
# ---------------------------

def remove_urls(text):
    """Remove all URLs from text."""
    if pd.isna(text):
        return ""
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

def remove_mentions(text):
    """Remove @mentions from text."""
    if pd.isna(text):
        return ""
    return re.sub(r'@\w+', '', text)

def remove_hashtag_symbol(text):
    """Remove # symbol but keep the word."""
    if pd.isna(text):
        return ""
    return re.sub(r'#(\w+)', r'\1', text)

def remove_hashtags_completely(text):
    """Remove hashtags completely including the word."""
    if pd.isna(text):
        return ""
    return re.sub(r'#\w+', '', text)

def remove_emojis(text):
    """Remove emojis from text."""
    if pd.isna(text):
        return ""
    if EMOJI_AVAILABLE:
        return emoji.replace_emoji(text, replace='')
    else:
        # Fallback: remove common emoji unicode ranges
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

def remove_special_characters(text):
    """Remove special characters, keep alphanumeric and basic punctuation."""
    if pd.isna(text):
        return ""
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

def remove_extra_whitespace(text):
    """Remove extra whitespace and normalize spaces."""
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_newlines(text):
    """Replace newlines with spaces."""
    if pd.isna(text):
        return ""
    return text.replace('\n', ' ').replace('\r', ' ')

def normalize_unicode(text):
    """Normalize unicode characters."""
    if pd.isna(text):
        return ""
    return unicodedata.normalize('NFKD', text)

def to_lowercase(text):
    """Convert text to lowercase."""
    if pd.isna(text):
        return ""
    return text.lower()

def remove_rt_prefix(text):
    """Remove 'RT' prefix from retweets."""
    if pd.isna(text):
        return ""
    return re.sub(r'^RT\s*:?\s*', '', text, flags=re.IGNORECASE)

def remove_numbers(text):
    """Remove all numbers from text."""
    if pd.isna(text):
        return ""
    return re.sub(r'\d+', '', text)

# ---------------------------
# Main Cleaning Pipeline
# ---------------------------

def clean_text(text, config=None):
    """Apply text cleaning pipeline based on configuration."""
    if config is None:
        config = {}
    
    defaults = {
        'remove_urls': True,
        'remove_mentions': True,
        'remove_hashtags': 'symbol',
        'remove_emojis': True,
        'remove_special_chars': True,
        'lowercase': True,
        'remove_numbers': False,
        'normalize_unicode': True,
        'remove_rt_prefix': True
    }
    
    config = {**defaults, **config}
    
    if pd.isna(text) or text == "":
        return ""
    
    text = remove_newlines(text)
    
    if config['remove_rt_prefix']:
        text = remove_rt_prefix(text)
    if config['remove_urls']:
        text = remove_urls(text)
    if config['remove_mentions']:
        text = remove_mentions(text)
    if config['remove_hashtags'] == 'symbol':
        text = remove_hashtag_symbol(text)
    elif config['remove_hashtags'] == 'complete':
        text = remove_hashtags_completely(text)
    if config['remove_emojis']:
        text = remove_emojis(text)
    if config['normalize_unicode']:
        text = normalize_unicode(text)
    if config['remove_special_chars']:
        text = remove_special_characters(text)
    if config['remove_numbers']:
        text = remove_numbers(text)
    if config['lowercase']:
        text = to_lowercase(text)
    
    text = remove_extra_whitespace(text)
    return text

# ---------------------------
# DataFrame Processing
# ---------------------------

def load_raw_data(filename):
    """Load raw CSV data from the raw data path."""
    filepath = os.path.join(RAW_DATA_PATH, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded {len(df)} rows from {filename}")
    return df

def process_dataframe(df, text_column='text', config=None):
    """Process a DataFrame with tweet data."""
    df = df.copy()
    
    print("Cleaning text...")
    df['text_cleaned'] = df[text_column].apply(lambda x: clean_text(x, config))
    df['text_length'] = df['text_cleaned'].str.len()
    df['word_count'] = df['text_cleaned'].str.split().str.len().fillna(0).astype(int)
    
    # Parse JSON columns if they exist
    json_columns = ['urls', 'hashtags', 'mentions', 'media']
    for col in json_columns:
        if col in df.columns:
            df[f'{col}_count'] = df[col].apply(
                lambda x: len(json.loads(x)) if pd.notna(x) and x else 0
            )
    
    # Convert created_at to datetime
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['date'] = df['created_at'].dt.date
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.day_name()
    
    # Calculate engagement score
    engagement_cols = ['like_count', 'retweet_count', 'comment_count']
    if all(col in df.columns for col in engagement_cols):
        df['total_engagement'] = df[engagement_cols].sum(axis=1)
    
    # Remove empty texts
    initial_count = len(df)
    df = df[df['text_cleaned'].str.len() > 0]
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed} rows with empty text")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text_cleaned'], keep='first')
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Removed {removed} duplicate tweets")
    
    print(f"Processed {len(df)} tweets")
    return df

def save_processed_data(df, filename, format='csv', append=False):
    """Save processed DataFrame with optional append mode."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    saved_files = []
    
    if format in ['csv', 'both']:
        csv_path = os.path.join(PROCESSED_DATA_PATH, f"{filename}.csv")
        
        if append and os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path, encoding='utf-8')
            df = pd.concat([existing_df, df], ignore_index=True)
            if 'tweet_id' in df.columns:
                df = df.drop_duplicates(subset=['tweet_id'], keep='last')
            else:
                df = df.drop_duplicates(subset=['text_cleaned'], keep='last')
            print(f"Appended to existing file. Total rows: {len(df)}")
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        saved_files.append(csv_path)
        print(f"Saved CSV to {csv_path}")
    
    if format in ['parquet', 'both']:
        parquet_path = os.path.join(PROCESSED_DATA_PATH, f"{filename}.parquet")
        df.to_parquet(parquet_path, index=False)
        saved_files.append(parquet_path)
        print(f"Saved Parquet to {parquet_path}")
    
    return saved_files

def get_processing_summary(df):
    """Generate a summary of the processed data."""
    summary = {
        'total_tweets': len(df),
        'date_range': None,
        'avg_text_length': df['text_length'].mean() if 'text_length' in df.columns else None,
        'avg_word_count': df['word_count'].mean() if 'word_count' in df.columns else None,
        'total_engagement': df['total_engagement'].sum() if 'total_engagement' in df.columns else None,
    }
    
    if 'created_at' in df.columns:
        summary['date_range'] = f"{df['created_at'].min()} to {df['created_at'].max()}"
    
    return summary

def process_file(input_filename, output_filename=None, config=None, save_format='csv', append=False):
    """Main function to process a single file."""
    if output_filename is None:
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_processed"
    
    df = load_raw_data(input_filename)
    df_processed = process_dataframe(df, config=config)
    saved_files = save_processed_data(df_processed, output_filename, format=save_format, append=append)
    summary = get_processing_summary(df_processed)
    
    return df_processed, summary, saved_files

# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Twitter Data Text Processor")
    print("=" * 60)
    
    if not os.path.exists(RAW_DATA_PATH):
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        print(f"Created directory: {RAW_DATA_PATH}")
        print("Please add raw CSV files and run again.")
        exit()
    
    csv_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in raw data directory")
        exit()
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    
    if len(csv_files) == 1:
        selected_file = csv_files[0]
    else:
        choice = input(f"\nSelect file (1-{len(csv_files)}): ").strip()
        selected_file = csv_files[int(choice) - 1]
    
    # Append option
    append_mode = input("\nAppend to existing processed file? (y/n, default n): ").lower() == 'y'
    
    print(f"\nProcessing: {selected_file}")
    df, summary, files = process_file(selected_file, append=append_mode)
    
    print(f"\nSummary: {summary['total_tweets']} tweets processed")
    print("Processing complete!")