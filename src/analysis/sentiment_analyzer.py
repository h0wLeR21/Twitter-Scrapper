import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import re

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (keep the text after #)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment_scores(self, text):
        """Get VADER sentiment scores"""
        scores = self.sia.polarity_scores(text)
        return scores
    
    def classify_sentiment(self, compound_score):
        """Classify sentiment based on compound score"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_csv(self, input_path, output_path, text_column='text'):
        """
        Analyze sentiment from CSV file
        
        Parameters:
        - input_path: Path to input CSV file
        - output_path: Path to save results CSV
        - text_column: Name of the column containing text to analyze
        """
        print(f"Reading data from: {input_path}")
        
        # Read CSV
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        print(f"Processing {len(df)} records...")
        
        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Get sentiment scores
        sentiment_scores = df['cleaned_text'].apply(self.get_sentiment_scores)
        
        # Extract individual scores
        df['neg_score'] = sentiment_scores.apply(lambda x: x['neg'])
        df['neu_score'] = sentiment_scores.apply(lambda x: x['neu'])
        df['pos_score'] = sentiment_scores.apply(lambda x: x['pos'])
        df['compound_score'] = sentiment_scores.apply(lambda x: x['compound'])
        
        # Classify sentiment
        df['sentiment'] = df['compound_score'].apply(self.classify_sentiment)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary statistics
        self.print_summary(df)
        
        return df
    
    def print_summary(self, df):
        """Print sentiment analysis summary"""
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*50)
        
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        
        print(f"\nTotal records analyzed: {total}")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.2f}%)")
        
        print("\nAverage Scores:")
        print(f"  Positive: {df['pos_score'].mean():.4f}")
        print(f"  Neutral: {df['neu_score'].mean():.4f}")
        print(f"  Negative: {df['neg_score'].mean():.4f}")
        print(f"  Compound: {df['compound_score'].mean():.4f}")
        print("="*50)

def main():
    # Define paths
    processed_dir = r"D:\code\Edure\New folder\Twitter-Scrapper\data\processed"
    results_dir = r"D:\code\Edure\New folder\Twitter-Scrapper\data\results"
    
    # Check if processed directory exists
    if not os.path.exists(processed_dir):
        print(f"Error: Directory not found: {processed_dir}")
        return
    
    # List all CSV files in the processed directory
    csv_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in: {processed_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) in processed directory:")
    for i, file in enumerate(csv_files, 1):
        file_path = os.path.join(processed_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # Size in KB
        print(f"  {i}. {file} ({file_size:.2f} KB)")
    
    # Let user choose which file to process
    if len(csv_files) == 1:
        choice = 0
        print(f"\nAutomatically selecting: {csv_files[0]}")
    else:
        while True:
            try:
                choice = int(input(f"\nSelect file number (1-{len(csv_files)}): ")) - 1
                if 0 <= choice < len(csv_files):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print("Please enter a valid number")
    
    selected_file = csv_files[choice]
    input_path = os.path.join(processed_dir, selected_file)
    
    # Create output filename based on input filename
    output_filename = selected_file.replace('_processed.csv', '_sentiment_results.csv')
    if output_filename == selected_file:
        output_filename = selected_file.replace('.csv', '_sentiment_results.csv')
    output_path = os.path.join(results_dir, output_filename)
    
    print(f"\nProcessing: {selected_file}")
    print(f"Output will be saved to: {output_filename}")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze sentiment
    # Change 'text' to match your actual column name if different
    df_results = analyzer.analyze_csv(input_path, output_path, text_column='text')
    
    print("\nSample results:")
    print(df_results[['text', 'cleaned_text', 'sentiment', 'compound_score']].head())

if __name__ == "__main__":
    main()