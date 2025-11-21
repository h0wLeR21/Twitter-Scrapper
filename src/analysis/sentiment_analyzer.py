import pandas as pd
import numpy as np
import os
import re

# NLTK imports with download handling
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def download_nltk_data():
    """Download required NLTK data."""
    resources = ['vader_lexicon', 'punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else resource)
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

download_nltk_data()

# ---------------------------
# Configuration
# ---------------------------

PROCESSED_DATA_PATH = r"D:\code\Edure\New folder\Twitter-Scrapper\data\processed"
RESULTS_DATA_PATH = r"D:\code\Edure\New folder\Twitter-Scrapper\data\results"

# ---------------------------
# Sentiment Analyzer Class
# ---------------------------

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
    
    def clean_text(self, text):
        """Clean and preprocess text for sentiment analysis."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment_scores(self, text):
        """Get VADER sentiment scores."""
        if not text or pd.isna(text):
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        return self.sia.polarity_scores(text)
    
    def classify_sentiment(self, compound_score):
        """Classify sentiment based on compound score."""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def get_sentiment_strength(self, compound_score):
        """Get sentiment strength category."""
        abs_score = abs(compound_score)
        if abs_score >= 0.6:
            return 'strong'
        elif abs_score >= 0.3:
            return 'moderate'
        elif abs_score >= 0.05:
            return 'weak'
        else:
            return 'neutral'
    
    def analyze_dataframe(self, df, text_column='text'):
        """Analyze sentiment for a DataFrame."""
        df = df.copy()
        
        # Use cleaned text if available, otherwise clean original
        if 'text_cleaned' in df.columns:
            analysis_col = 'text_cleaned'
        else:
            print("Creating cleaned text column...")
            df['text_cleaned'] = df[text_column].apply(self.clean_text)
            analysis_col = 'text_cleaned'
        
        print(f"Analyzing sentiment for {len(df)} records...")
        
        # Get sentiment scores
        sentiment_scores = df[analysis_col].apply(self.get_sentiment_scores)
        
        df['neg_score'] = sentiment_scores.apply(lambda x: x['neg'])
        df['neu_score'] = sentiment_scores.apply(lambda x: x['neu'])
        df['pos_score'] = sentiment_scores.apply(lambda x: x['pos'])
        df['compound_score'] = sentiment_scores.apply(lambda x: x['compound'])
        
        # Classify sentiment
        df['sentiment'] = df['compound_score'].apply(self.classify_sentiment)
        df['sentiment_strength'] = df['compound_score'].apply(self.get_sentiment_strength)
        
        return df
    
    def analyze_csv(self, input_path, output_path, text_column='text'):
        """Analyze sentiment from CSV file and save results."""
        print(f"Reading data from: {input_path}")
        
        df = pd.read_csv(input_path, encoding='utf-8')
        
        if text_column not in df.columns and 'text_cleaned' not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        df = self.analyze_dataframe(df, text_column)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results saved to: {output_path}")
        
        self.print_summary(df)
        
        return df
    
    def print_summary(self, df):
        """Print sentiment analysis summary."""
        print("\n" + "=" * 50)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("=" * 50)
        
        total = len(df)
        print(f"\nTotal records analyzed: {total}")
        
        # Sentiment distribution
        print("\nSentiment Distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        for sentiment in ['positive', 'neutral', 'negative']:
            count = sentiment_counts.get(sentiment, 0)
            pct = (count / total) * 100
            bar = '█' * int(pct / 2)
            print(f"  {sentiment.capitalize():10} {count:6} ({pct:5.1f}%) {bar}")
        
        # Strength distribution
        print("\nSentiment Strength:")
        strength_counts = df['sentiment_strength'].value_counts()
        for strength in ['strong', 'moderate', 'weak', 'neutral']:
            count = strength_counts.get(strength, 0)
            pct = (count / total) * 100
            print(f"  {strength.capitalize():10} {count:6} ({pct:5.1f}%)")
        
        # Average scores
        print("\nAverage Scores:")
        print(f"  Positive:  {df['pos_score'].mean():.4f}")
        print(f"  Neutral:   {df['neu_score'].mean():.4f}")
        print(f"  Negative:  {df['neg_score'].mean():.4f}")
        print(f"  Compound:  {df['compound_score'].mean():.4f}")
        
        # Most positive and negative
        print("\nExtreme Examples:")
        most_positive = df.loc[df['compound_score'].idxmax()]
        most_negative = df.loc[df['compound_score'].idxmin()]
        
        text_col = 'text_cleaned' if 'text_cleaned' in df.columns else 'text'
        print(f"  Most positive ({most_positive['compound_score']:.3f}):")
        print(f"    \"{most_positive[text_col][:80]}...\"")
        print(f"  Most negative ({most_negative['compound_score']:.3f}):")
        print(f"    \"{most_negative[text_col][:80]}...\"")
        
        print("=" * 50)
    
    def get_summary_dict(self, df):
        """Return summary as dictionary for plotting."""
        total = len(df)
        sentiment_counts = df['sentiment'].value_counts()
        
        return {
            'total': total,
            'positive': sentiment_counts.get('positive', 0),
            'neutral': sentiment_counts.get('neutral', 0),
            'negative': sentiment_counts.get('negative', 0),
            'avg_compound': df['compound_score'].mean(),
            'avg_positive': df['pos_score'].mean(),
            'avg_negative': df['neg_score'].mean(),
            'avg_neutral': df['neu_score'].mean(),
        }

# ---------------------------
# Main
# ---------------------------

def main():
    print("=" * 60)
    print("Twitter Sentiment Analyzer")
    print("=" * 60)
    
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"Error: Directory not found: {PROCESSED_DATA_PATH}")
        return
    
    csv_files = [f for f in os.listdir(PROCESSED_DATA_PATH) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in: {PROCESSED_DATA_PATH}")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, file in enumerate(csv_files, 1):
        file_path = os.path.join(PROCESSED_DATA_PATH, file)
        size_kb = os.path.getsize(file_path) / 1024
        print(f"  {i}. {file} ({size_kb:.1f} KB)")
    
    if len(csv_files) == 1:
        choice = 0
    else:
        choice = int(input(f"\nSelect file (1-{len(csv_files)}): ")) - 1
    
    selected_file = csv_files[choice]
    input_path = os.path.join(PROCESSED_DATA_PATH, selected_file)
    
    # Generate output filename
    output_filename = selected_file.replace('_processed.csv', '_sentiment.csv')
    if output_filename == selected_file:
        output_filename = selected_file.replace('.csv', '_sentiment.csv')
    output_path = os.path.join(RESULTS_DATA_PATH, output_filename)
    
    print(f"\nProcessing: {selected_file}")
    
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_csv(input_path, output_path, text_column='text')
    
    print("\nSample results:")
    display_cols = ['text_cleaned', 'sentiment', 'compound_score']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].head(10).to_string())

if __name__ == "__main__":
    main()