import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import Counter
import re

# ---------------------------
# Configuration
# ---------------------------

RESULTS_DATA_PATH = r"D:\code\Edure\New folder\Twitter-Scrapper\data\results"
PLOTS_PATH = r"D:\code\Edure\New folder\Twitter-Scrapper\src\visualization\plots"

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'positive': '#2ecc71',
    'neutral': '#3498db', 
    'negative': '#e74c3c',
    'primary': '#9b59b6',
    'secondary': '#1abc9c'
}

# ---------------------------
# Plot Generator Class
# ---------------------------

class PlotGenerator:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or PLOTS_PATH
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_plot(self, fig, filename):
        """Save plot to output directory."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {filename}")
        return filepath
    
    # ---------------------------
    # Sentiment Plots
    # ---------------------------
    
    def plot_sentiment_distribution(self, df, filename='sentiment_distribution.png'):
        """Pie chart of sentiment distribution."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        counts = df['sentiment'].value_counts()
        colors = [COLORS.get(s, '#95a5a6') for s in counts.index]
        
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index.str.capitalize(),
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.02] * len(counts),
            shadow=True,
            startangle=90
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        
        total = len(df)
        ax.text(0, -1.3, f'Total: {total:,} tweets', ha='center', fontsize=11)
        
        return self.save_plot(fig, filename)
    
    def plot_sentiment_bar(self, df, filename='sentiment_bar.png'):
        """Bar chart of sentiment counts."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        counts = df['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative'])
        colors = [COLORS[s] for s in counts.index]
        
        bars = ax.bar(counts.index.str.capitalize(), counts.values, color=colors, edgecolor='white', linewidth=2)
        
        for bar, count in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{count:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return self.save_plot(fig, filename)
    
    def plot_compound_score_distribution(self, df, filename='compound_distribution.png'):
        """Histogram of compound scores."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(df['compound_score'], bins=50, color=COLORS['primary'], 
                edgecolor='white', alpha=0.8)
        
        ax.axvline(x=0.05, color=COLORS['positive'], linestyle='--', 
                  linewidth=2, label='Positive threshold')
        ax.axvline(x=-0.05, color=COLORS['negative'], linestyle='--', 
                  linewidth=2, label='Negative threshold')
        ax.axvline(x=df['compound_score'].mean(), color='black', linestyle='-', 
                  linewidth=2, label=f'Mean ({df["compound_score"].mean():.3f})')
        
        ax.set_xlabel('Compound Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Compound Sentiment Scores', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return self.save_plot(fig, filename)
    
    def plot_sentiment_over_time(self, df, filename='sentiment_timeline.png'):
        """Line plot of sentiment over time."""
        if 'created_at' not in df.columns and 'date' not in df.columns:
            print("No date column found. Skipping timeline plot.")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['created_at']).dt.date
        
        daily = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in daily.columns:
                ax.plot(daily.index, daily[sentiment], 
                       color=COLORS[sentiment], linewidth=2, 
                       marker='o', markersize=4, label=sentiment.capitalize())
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Tweet Count', fontsize=12)
        ax.set_title('Sentiment Trend Over Time', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        plt.xticks(rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return self.save_plot(fig, filename)
    
    def plot_hourly_sentiment(self, df, filename='hourly_sentiment.png'):
        """Heatmap of sentiment by hour."""
        if 'hour' not in df.columns and 'created_at' not in df.columns:
            print("No hour data found. Skipping hourly plot.")
            return None
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['created_at']).dt.hour
        
        hourly = df.groupby(['hour', 'sentiment']).size().unstack(fill_value=0)
        hourly_pct = hourly.div(hourly.sum(axis=1), axis=0) * 100
        
        x = range(24)
        width = 0.25
        
        for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
            if sentiment in hourly_pct.columns:
                values = [hourly_pct[sentiment].get(h, 0) for h in x]
                ax.bar([h + i*width for h in x], values, width, 
                      color=COLORS[sentiment], label=sentiment.capitalize())
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_title('Sentiment Distribution by Hour', fontsize=16, fontweight='bold')
        ax.set_xticks([h + width for h in x])
        ax.set_xticklabels([f'{h:02d}:00' for h in x], rotation=45)
        ax.legend(loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return self.save_plot(fig, filename)
    
    # ---------------------------
    # Engagement Plots
    # ---------------------------
    
    def plot_engagement_by_sentiment(self, df, filename='engagement_sentiment.png'):
        """Box plot of engagement by sentiment."""
        if 'total_engagement' not in df.columns:
            eng_cols = ['like_count', 'retweet_count', 'comment_count']
            available = [c for c in eng_cols if c in df.columns]
            if not available:
                print("No engagement columns found. Skipping.")
                return None
            df['total_engagement'] = df[available].sum(axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        order = ['positive', 'neutral', 'negative']
        colors = [COLORS[s] for s in order]
        
        bp = ax.boxplot([df[df['sentiment'] == s]['total_engagement'].values for s in order],
                       labels=[s.capitalize() for s in order], patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Sentiment', fontsize=12)
        ax.set_ylabel('Total Engagement', fontsize=12)
        ax.set_title('Engagement Distribution by Sentiment', fontsize=16, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return self.save_plot(fig, filename)
    
    def plot_engagement_scatter(self, df, filename='engagement_scatter.png'):
        """Scatter plot of compound score vs engagement."""
        if 'total_engagement' not in df.columns:
            print("No engagement data. Skipping scatter plot.")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = df['sentiment'].map(COLORS)
        
        scatter = ax.scatter(df['compound_score'], df['total_engagement'],
                           c=colors, alpha=0.5, s=30, edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Compound Sentiment Score', fontsize=12)
        ax.set_ylabel('Total Engagement', fontsize=12)
        ax.set_title('Sentiment vs Engagement', fontsize=16, fontweight='bold')
        
        # Legend
        handles = [plt.scatter([], [], c=COLORS[s], label=s.capitalize(), s=100) 
                  for s in ['positive', 'neutral', 'negative']]
        ax.legend(handles=handles, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return self.save_plot(fig, filename)
    
    # ---------------------------
    # Word Analysis Plots
    # ---------------------------
    
    def plot_word_frequency(self, df, filename='word_frequency.png', top_n=20):
        """Bar chart of most common words."""
        text_col = 'text_cleaned' if 'text_cleaned' in df.columns else 'text'
        
        all_words = ' '.join(df[text_col].dropna()).lower().split()
        
        # Remove short words
        all_words = [w for w in all_words if len(w) > 2]
        
        word_counts = Counter(all_words).most_common(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        words, counts = zip(*word_counts)
        y_pos = range(len(words))
        
        bars = ax.barh(y_pos, counts, color=COLORS['primary'], edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_title(f'Top {top_n} Most Common Words', fontsize=16, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        return self.save_plot(fig, filename)
    
    def plot_word_frequency_by_sentiment(self, df, filename='words_by_sentiment.png', top_n=10):
        """Word frequency split by sentiment."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        text_col = 'text_cleaned' if 'text_cleaned' in df.columns else 'text'
        
        for ax, sentiment in zip(axes, ['positive', 'neutral', 'negative']):
            subset = df[df['sentiment'] == sentiment]
            words = ' '.join(subset[text_col].dropna()).lower().split()
            words = [w for w in words if len(w) > 2]
            
            word_counts = Counter(words).most_common(top_n)
            
            if word_counts:
                words_list, counts = zip(*word_counts)
                y_pos = range(len(words_list))
                
                ax.barh(y_pos, counts, color=COLORS[sentiment], edgecolor='white')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(words_list)
                ax.invert_yaxis()
            
            ax.set_title(f'{sentiment.capitalize()}', fontsize=14, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        fig.suptitle('Top Words by Sentiment', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self.save_plot(fig, filename)
    
    # ---------------------------
    # Summary Dashboard
    # ---------------------------
    
    def create_dashboard(self, df, filename='dashboard.png'):
        """Create a comprehensive dashboard."""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Sentiment pie chart
        ax1 = fig.add_subplot(2, 3, 1)
        counts = df['sentiment'].value_counts()
        colors = [COLORS.get(s, '#95a5a6') for s in counts.index]
        ax1.pie(counts.values, labels=counts.index.str.capitalize(),
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Sentiment Distribution', fontweight='bold')
        
        # 2. Compound score histogram
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(df['compound_score'], bins=30, color=COLORS['primary'], edgecolor='white')
        ax2.axvline(x=df['compound_score'].mean(), color='red', linestyle='--', linewidth=2)
        ax2.set_title('Compound Score Distribution', fontweight='bold')
        ax2.set_xlabel('Compound Score')
        
        # 3. Sentiment strength
        ax3 = fig.add_subplot(2, 3, 3)
        if 'sentiment_strength' in df.columns:
            strength = df['sentiment_strength'].value_counts()
            ax3.bar(strength.index.str.capitalize(), strength.values, 
                   color=COLORS['secondary'], edgecolor='white')
            ax3.set_title('Sentiment Strength', fontweight='bold')
        
        # 4. Timeline (if dates available)
        ax4 = fig.add_subplot(2, 3, 4)
        if 'date' in df.columns or 'created_at' in df.columns:
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['created_at']).dt.date
            daily = df.groupby('date').size()
            ax4.plot(daily.index, daily.values, color=COLORS['primary'], linewidth=2)
            ax4.fill_between(daily.index, daily.values, alpha=0.3, color=COLORS['primary'])
            ax4.set_title('Tweet Volume Over Time', fontweight='bold')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. Score components
        ax5 = fig.add_subplot(2, 3, 5)
        scores = ['pos_score', 'neu_score', 'neg_score']
        if all(s in df.columns for s in scores):
            means = [df[s].mean() for s in scores]
            labels = ['Positive', 'Neutral', 'Negative']
            colors_list = [COLORS['positive'], COLORS['neutral'], COLORS['negative']]
            ax5.bar(labels, means, color=colors_list, edgecolor='white')
            ax5.set_title('Average Score Components', fontweight='bold')
            ax5.set_ylabel('Average Score')
        
        # 6. Stats text box
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        stats_text = f"""
        SUMMARY STATISTICS
        {'='*30}
        
        Total Tweets: {len(df):,}
        
        Sentiment Breakdown:
          • Positive: {(df['sentiment']=='positive').sum():,} ({(df['sentiment']=='positive').mean()*100:.1f}%)
          • Neutral: {(df['sentiment']=='neutral').sum():,} ({(df['sentiment']=='neutral').mean()*100:.1f}%)
          • Negative: {(df['sentiment']=='negative').sum():,} ({(df['sentiment']=='negative').mean()*100:.1f}%)
        
        Average Scores:
          • Compound: {df['compound_score'].mean():.4f}
          • Positive: {df['pos_score'].mean():.4f}
          • Negative: {df['neg_score'].mean():.4f}
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Twitter Sentiment Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        return self.save_plot(fig, filename)
    
    # ---------------------------
    # Generate All Plots
    # ---------------------------
    
    def generate_all_plots(self, df, prefix=''):
        """Generate all available plots."""
        print("\nGenerating plots...")
        
        plots = []
        
        plots.append(self.plot_sentiment_distribution(df, f'{prefix}sentiment_pie.png'))
        plots.append(self.plot_sentiment_bar(df, f'{prefix}sentiment_bar.png'))
        plots.append(self.plot_compound_score_distribution(df, f'{prefix}compound_dist.png'))
        plots.append(self.plot_sentiment_over_time(df, f'{prefix}timeline.png'))
        plots.append(self.plot_hourly_sentiment(df, f'{prefix}hourly.png'))
        plots.append(self.plot_engagement_by_sentiment(df, f'{prefix}engagement_box.png'))
        plots.append(self.plot_engagement_scatter(df, f'{prefix}engagement_scatter.png'))
        plots.append(self.plot_word_frequency(df, f'{prefix}word_freq.png'))
        plots.append(self.plot_word_frequency_by_sentiment(df, f'{prefix}words_sentiment.png'))
        plots.append(self.create_dashboard(df, f'{prefix}dashboard.png'))
        
        plots = [p for p in plots if p is not None]
        print(f"\nGenerated {len(plots)} plots in: {self.output_dir}")
        
        return plots

# ---------------------------
# Main
# ---------------------------

def main():
    print("=" * 60)
    print("Twitter Sentiment Plot Generator")
    print("=" * 60)
    
    if not os.path.exists(RESULTS_DATA_PATH):
        print(f"Error: Results directory not found: {RESULTS_DATA_PATH}")
        return
    
    csv_files = [f for f in os.listdir(RESULTS_DATA_PATH) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in: {RESULTS_DATA_PATH}")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s):")
    for i, file in enumerate(csv_files, 1):
        print(f"  {i}. {file}")
    
    if len(csv_files) == 1:
        choice = 0
    else:
        choice = int(input(f"\nSelect file (1-{len(csv_files)}): ")) - 1
    
    selected_file = csv_files[choice]
    filepath = os.path.join(RESULTS_DATA_PATH, selected_file)
    
    print(f"\nLoading: {selected_file}")
    df = pd.read_csv(filepath, encoding='utf-8')
    print(f"Loaded {len(df)} records")
    
    # Check required columns
    if 'sentiment' not in df.columns or 'compound_score' not in df.columns:
        print("Error: File must contain 'sentiment' and 'compound_score' columns.")
        print("Please run sentiment_analyzer.py first.")
        return
    
    # Generate plots
    prefix = selected_file.replace('.csv', '_').replace('_sentiment', '')
    plotter = PlotGenerator()
    plotter.generate_all_plots(df, prefix=prefix)
    
    print("\nPlot generation complete!")

if __name__ == "__main__":
    main()