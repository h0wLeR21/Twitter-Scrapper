import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class SentimentPlotGenerator:
    def __init__(self):
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def load_data(self, file_path):
        """Load sentiment results from CSV"""
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        
        # Convert date column if exists
        date_columns = ['date', 'created_at', 'timestamp', 'Date', 'Created_at']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    df['date'] = df[col]
                    break
                except:
                    continue
        
        print(f"Loaded {len(df)} records")
        return df
    
    def plot_sentiment_distribution(self, df, output_path):
        """Plot sentiment distribution pie chart and bar chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        sentiment_counts = df['sentiment'].value_counts()
        colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        plot_colors = [colors.get(sent, '#3498db') for sent in sentiment_counts.index]
        
        # Pie chart
        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=plot_colors, startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'})
        ax1.set_title('Sentiment Distribution', fontsize=14, weight='bold', pad=20)
        
        # Bar chart
        bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=plot_colors, alpha=0.8)
        ax2.set_title('Sentiment Counts', fontsize=14, weight='bold', pad=20)
        ax2.set_xlabel('Sentiment', fontsize=12, weight='bold')
        ax2.set_ylabel('Count', fontsize=12, weight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, weight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_sentiment_scores(self, df, output_path):
        """Plot distribution of sentiment scores"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        scores = ['pos_score', 'neu_score', 'neg_score', 'compound_score']
        titles = ['Positive Scores', 'Neutral Scores', 'Negative Scores', 'Compound Scores']
        colors_list = ['#2ecc71', '#95a5a6', '#e74c3c', '#3498db']
        
        for idx, (score, title, color) in enumerate(zip(scores, titles, colors_list)):
            ax = axes[idx // 2, idx % 2]
            
            # Histogram
            ax.hist(df[score], bins=30, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(title, fontsize=14, weight='bold', pad=15)
            ax.set_xlabel('Score', fontsize=11, weight='bold')
            ax.set_ylabel('Frequency', fontsize=11, weight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean line
            mean_val = df[score].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_sentiment_over_time(self, df, output_path):
        """Plot sentiment trends over time"""
        if 'date' not in df.columns:
            print("No date column found, skipping time series plot")
            return
        
        df_time = df.copy()
        df_time = df_time.sort_values('date')
        df_time['date_only'] = df_time['date'].dt.date
        
        # Sentiment counts over time
        sentiment_by_date = df_time.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Stacked area chart
        colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        sentiment_by_date.plot(kind='area', stacked=True, ax=ax1, 
                              color=[colors.get(col, '#3498db') for col in sentiment_by_date.columns],
                              alpha=0.7)
        ax1.set_title('Sentiment Trends Over Time (Stacked)', fontsize=14, weight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12, weight='bold')
        ax1.set_ylabel('Number of Tweets', fontsize=12, weight='bold')
        ax1.legend(title='Sentiment', fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Line chart for compound score
        compound_by_date = df_time.groupby('date_only')['compound_score'].mean()
        ax2.plot(compound_by_date.index, compound_by_date.values, 
                marker='o', linewidth=2, markersize=4, color='#3498db')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(compound_by_date.index, compound_by_date.values, 0, 
                        where=(compound_by_date.values > 0), alpha=0.3, color='#2ecc71', label='Positive')
        ax2.fill_between(compound_by_date.index, compound_by_date.values, 0, 
                        where=(compound_by_date.values < 0), alpha=0.3, color='#e74c3c', label='Negative')
        ax2.set_title('Average Compound Score Over Time', fontsize=14, weight='bold', pad=20)
        ax2.set_xlabel('Date', fontsize=12, weight='bold')
        ax2.set_ylabel('Average Compound Score', fontsize=12, weight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_word_length_sentiment(self, df, output_path):
        """Plot relationship between text length and sentiment"""
        df['text_length'] = df['text'].astype(str).apply(len)
        df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot for word count by sentiment
        sentiment_order = ['negative', 'neutral', 'positive']
        colors = {'positive': '#2ecc71', 'neutral': '#95a5a6', 'negative': '#e74c3c'}
        
        box_data = [df[df['sentiment'] == sent]['word_count'].values 
                   for sent in sentiment_order if sent in df['sentiment'].values]
        box_colors = [colors[sent] for sent in sentiment_order if sent in df['sentiment'].values]
        
        bp = ax1.boxplot(box_data, labels=[s.capitalize() for s in sentiment_order if s in df['sentiment'].values],
                        patch_artist=True, showmeans=True)
        
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Word Count Distribution by Sentiment', fontsize=14, weight='bold', pad=20)
        ax1.set_xlabel('Sentiment', fontsize=12, weight='bold')
        ax1.set_ylabel('Word Count', fontsize=12, weight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Scatter plot: word count vs compound score
        sentiment_colors = df['sentiment'].map(colors)
        ax2.scatter(df['word_count'], df['compound_score'], 
                   c=sentiment_colors, alpha=0.5, s=30)
        ax2.set_title('Word Count vs Compound Score', fontsize=14, weight='bold', pad=20)
        ax2.set_xlabel('Word Count', fontsize=12, weight='bold')
        ax2.set_ylabel('Compound Score', fontsize=12, weight='bold')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[sent], alpha=0.7, label=sent.capitalize()) 
                          for sent in ['positive', 'neutral', 'negative']]
        ax2.legend(handles=legend_elements, title='Sentiment', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_correlation_heatmap(self, df, output_path):
        """Plot correlation heatmap of sentiment scores"""
        score_cols = ['pos_score', 'neu_score', 'neg_score', 'compound_score']
        
        if 'word_count' in df.columns:
            score_cols.append('word_count')
        
        corr_matrix = df[score_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Sentiment Scores', fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all_plots(self, csv_file):
        """Generate all visualization plots"""
        # Load data
        df = self.load_data(csv_file)
        
        # Create output directory
        results_dir = os.path.dirname(csv_file)
        plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract username from filename
        filename = os.path.basename(csv_file)
        username = filename.split('_')[0] if '_' in filename else 'user'
        
        print(f"\nGenerating plots for {username}...\n")
        
        # Generate plots
        self.plot_sentiment_distribution(
            df, 
            os.path.join(plots_dir, f'{username}_sentiment_distribution.png')
        )
        
        self.plot_sentiment_scores(
            df, 
            os.path.join(plots_dir, f'{username}_sentiment_scores.png')
        )
        
        self.plot_sentiment_over_time(
            df, 
            os.path.join(plots_dir, f'{username}_sentiment_timeline.png')
        )
        
        self.plot_word_length_sentiment(
            df, 
            os.path.join(plots_dir, f'{username}_word_analysis.png')
        )
        
        self.plot_correlation_heatmap(
            df, 
            os.path.join(plots_dir, f'{username}_correlation_heatmap.png')
        )
        
        print(f"\n{'='*50}")
        print(f"All plots saved to: {plots_dir}")
        print(f"{'='*50}")

def main():
    results_dir = r"D:\code\Edure\New folder\Twitter-Scrapper\data\results"
    
    # Check if directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        return
    
    # List all CSV files
    csv_files = [f for f in os.listdir(results_dir) 
                if f.endswith('.csv') and 'sentiment' in f.lower()]
    
    if not csv_files:
        print(f"No sentiment result CSV files found in: {results_dir}")
        return
    
    print(f"Found {len(csv_files)} sentiment result file(s):")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    # Select file
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
    
    selected_file = os.path.join(results_dir, csv_files[choice])
    
    # Generate plots
    generator = SentimentPlotGenerator()
    generator.generate_all_plots(selected_file)

if __name__ == "__main__":
    main()