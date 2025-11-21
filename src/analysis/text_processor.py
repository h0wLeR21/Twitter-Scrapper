import pandas as pd
import numpy as np
import re
import os
import json
from datetime import datetime
import unicodedata
import emoji

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
    return emoji.replace_emoji(text, replace='')


def remove_special_characters(text):
    """Remove special characters, keep alphanumeric and basic punctuation."""
    if pd.isna(text):
        return ""
    # Keep letters, numbers, spaces, and basic punctuation
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
    """
    Apply text cleaning pipeline based on configuration.
    
    Config options:
        - remove_urls: bool (default: True)
        - remove_mentions: bool (default: True)
        - remove_hashtags: 'symbol' | 'complete' | 'none' (default: 'symbol')
        - remove_emojis: bool (default: True)
        - remove_special_chars: bool (default: True)
        - lowercase: bool (default: True)
        - remove_numbers: bool (default: False)
        - normalize_unicode: bool (default: True)
    """
    if config is None:
        config = {}
    
    # Default configuration
    defaults = {
        'remove_urls': True,
        'remove_mentions': True,
        'remove_hashtags': 'symbol',  # 'symbol', 'complete', or 'none'
        'remove_emojis': True,
        'remove_special_chars': True,
        'lowercase': True,
        'remove_numbers': False,
        'normalize_unicode': True,
        'remove_rt_prefix': True
    }
    
    # Merge with defaults
    config = {**defaults, **config}
    
    if pd.isna(text) or text == "":
        return ""
    
    # Apply cleaning steps in order
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
    
    # Always remove extra whitespace at the end
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
    print(f"✔ Loaded {len(df)} rows from {filename}")
    return df


def process_dataframe(df, text_column='text', config=None):
    """
    Process a DataFrame with tweet data.
    
    Args:
        df: pandas DataFrame
        text_column: name of the column containing text
        config: cleaning configuration dict
    
    Returns:
        Processed DataFrame
    """
    df = df.copy()
    
    # Clean the text column
    print("Cleaning text...")
    df['text_cleaned'] = df[text_column].apply(lambda x: clean_text(x, config))
    
    # Calculate text length
    df['text_length'] = df['text_cleaned'].str.len()
    df['word_count'] = df['text_cleaned'].str.split().str.len().fillna(0).astype(int)
    
    # Parse JSON columns if they exist
    json_columns = ['urls', 'hashtags', 'mentions', 'media']
    for col in json_columns:
        if col in df.columns:
            df[f'{col}_count'] = df[col].apply(lambda x: len(json.loads(x)) if pd.notna(x) and x else 0)
    
    # Convert created_at to datetime if exists
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['date'] = df['created_at'].dt.date
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.day_name()
    
    # Calculate engagement score
    engagement_cols = ['like_count', 'retweet_count', 'comment_count']
    if all(col in df.columns for col in engagement_cols):
        df['total_engagement'] = df[engagement_cols].sum(axis=1)
    
    # Remove empty texts after cleaning
    initial_count = len(df)
    df = df[df['text_cleaned'].str.len() > 0]
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"⚠ Removed {removed_count} rows with empty text after cleaning")
    
    # Remove duplicates based on cleaned text
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text_cleaned'], keep='first')
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"⚠ Removed {removed_count} duplicate tweets")
    
    print(f"✔ Processed {len(df)} tweets")
    return df


def save_processed_data(df, filename, format='csv'):
    """
    Save processed DataFrame to the processed data path.
    
    Args:
        df: pandas DataFrame
        filename: output filename (without extension)
        format: 'csv', 'parquet', or 'both'
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    
    saved_files = []
    
    if format in ['csv', 'both']:
        csv_path = os.path.join(PROCESSED_DATA_PATH, f"{filename}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        saved_files.append(csv_path)
        print(f"✔ Saved CSV to {csv_path}")
    
    if format in ['parquet', 'both']:
        parquet_path = os.path.join(PROCESSED_DATA_PATH, f"{filename}.parquet")
        df.to_parquet(parquet_path, index=False)
        saved_files.append(parquet_path)
        print(f"✔ Saved Parquet to {parquet_path}")
    
    return saved_files


def get_processing_summary(df):
    """Generate a summary of the processed data."""
    summary = {
        'total_tweets': len(df),
        'date_range': None,
        'avg_text_length': df['text_length'].mean() if 'text_length' in df.columns else None,
        'avg_word_count': df['word_count'].mean() if 'word_count' in df.columns else None,
        'total_engagement': df['total_engagement'].sum() if 'total_engagement' in df.columns else None,
        'languages': df['lang'].value_counts().to_dict() if 'lang' in df.columns else None,
        'tweet_types': {
            'original': len(df[(df['is_retweet'] == False) & (df['is_quote'] == False) & (df['is_reply'] == False)]) if all(col in df.columns for col in ['is_retweet', 'is_quote', 'is_reply']) else None,
            'retweets': df['is_retweet'].sum() if 'is_retweet' in df.columns else None,
            'quotes': df['is_quote'].sum() if 'is_quote' in df.columns else None,
            'replies': df['is_reply'].sum() if 'is_reply' in df.columns else None
        }
    }
    
    if 'created_at' in df.columns:
        summary['date_range'] = f"{df['created_at'].min()} to {df['created_at'].max()}"
    
    return summary


# ---------------------------
# Main
# ---------------------------

def process_file(input_filename, output_filename=None, config=None, save_format='csv'):
    """
    Main function to process a single file.
    
    Args:
        input_filename: name of the raw CSV file
        output_filename: name for the output file (without extension)
        config: cleaning configuration
        save_format: 'csv', 'parquet', or 'both'
    """
    if output_filename is None:
        # Generate output filename from input
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_processed"
    
    # Load data
    df = load_raw_data(input_filename)
    
    # Process
    df_processed = process_dataframe(df, config=config)
    
    # Save
    saved_files = save_processed_data(df_processed, output_filename, format=save_format)
    
    # Summary
    summary = get_processing_summary(df_processed)
    
    return df_processed, summary, saved_files


if __name__ == "__main__":
    print("=" * 60)
    print("Twitter Data Text Processor")
    print("=" * 60)
    
    # List available raw files
    print(f"\nLooking for raw files in: {RAW_DATA_PATH}")
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"⚠ Raw data directory does not exist: {RAW_DATA_PATH}")
        print("Creating directory...")
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        print("Please add raw CSV files to the directory and run again.")
        exit()
    
    csv_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]
    
    if not csv_files:
        print("⚠ No CSV files found in raw data directory")
        exit()
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    
    # Select file
    if len(csv_files) == 1:
        selected_file = csv_files[0]
        print(f"\nAuto-selecting: {selected_file}")
    else:
        choice = input("\nEnter file number to process (or 'all' for batch processing): ").strip()
        
        if choice.lower() == 'all':
            # Batch process all files
            print("\nBatch processing all files...")
            for f in csv_files:
                print(f"\n{'─' * 40}")
                print(f"Processing: {f}")
                try:
                    df, summary, files = process_file(f)
                    print(f"Summary: {summary['total_tweets']} tweets processed")
                except Exception as e:
                    print(f"✘ Error processing {f}: {e}")
            print(f"\n{'=' * 60}")
            print("Batch processing complete!")
            exit()
        else:
            try:
                idx = int(choice) - 1
                selected_file = csv_files[idx]
            except (ValueError, IndexError):
                print("Invalid selection")
                exit()
    
    # Configuration options
    print("\nCleaning Configuration:")
    print("  1. Standard (recommended) - URLs, mentions, hashtag symbols, emojis removed")
    print("  2. Minimal - Only URLs removed")
    print("  3. Aggressive - Everything removed, lowercase")
    print("  4. Custom - Configure each option")
    
    config_choice = input("\nSelect configuration (default: 1): ").strip() or "1"
    
    config = None
    if config_choice == "2":
        config = {
            'remove_urls': True,
            'remove_mentions': False,
            'remove_hashtags': 'none',
            'remove_emojis': False,
            'remove_special_chars': False,
            'lowercase': False,
            'remove_numbers': False
        }
    elif config_choice == "3":
        config = {
            'remove_urls': True,
            'remove_mentions': True,
            'remove_hashtags': 'complete',
            'remove_emojis': True,
            'remove_special_chars': True,
            'lowercase': True,
            'remove_numbers': True
        }
    elif config_choice == "4":
        config = {
            'remove_urls': input("Remove URLs? (y/n, default y): ").lower() != 'n',
            'remove_mentions': input("Remove @mentions? (y/n, default y): ").lower() != 'n',
            'remove_hashtags': input("Hashtags - 'symbol'/'complete'/'none' (default symbol): ").strip() or 'symbol',
            'remove_emojis': input("Remove emojis? (y/n, default y): ").lower() != 'n',
            'remove_special_chars': input("Remove special chars? (y/n, default y): ").lower() != 'n',
            'lowercase': input("Convert to lowercase? (y/n, default y): ").lower() != 'n',
            'remove_numbers': input("Remove numbers? (y/n, default n): ").lower() == 'y'
        }
    
    # Output format
    save_format = input("\nOutput format - 'csv'/'parquet'/'both' (default csv): ").strip() or 'csv'
    
    # Process
    print(f"\n{'─' * 40}")
    print(f"Processing: {selected_file}")
    print(f"{'─' * 40}\n")
    
    try:
        df, summary, files = process_file(selected_file, config=config, save_format=save_format)
        
        print(f"\n{'=' * 60}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total tweets: {summary['total_tweets']}")
        if summary['date_range']:
            print(f"Date range: {summary['date_range']}")
        if summary['avg_text_length']:
            print(f"Avg text length: {summary['avg_text_length']:.1f} chars")
        if summary['avg_word_count']:
            print(f"Avg word count: {summary['avg_word_count']:.1f} words")
        if summary['total_engagement']:
            print(f"Total engagement: {summary['total_engagement']:,}")
        
        print(f"\nTweet types:")
        for k, v in summary['tweet_types'].items():
            if v is not None:
                print(f"  - {k}: {v}")
        
        if summary['languages']:
            print(f"\nTop languages:")
            for lang, count in list(summary['languages'].items())[:5]:
                print(f"  - {lang}: {count}")
        
        print(f"\n{'=' * 60}")
        print("✔ Processing complete!")
        print(f"{'=' * 60}")
        
    except Exception as e:
        print(f"\n✘ Error: {e}")
        import traceback
        traceback.print_exc()