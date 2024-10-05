import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate social media data with sentiment labels
np.random.seed(42)

# Sample social media posts related to different topics/brands
data_size = 500
social_media_data = pd.DataFrame({
    'post': ['Post ' + str(i) for i in range(data_size)],
    'sentiment': np.random.choice(['positive', 'neutral', 'negative'], size=data_size, p=[0.4, 0.3, 0.3]),
    'hashtag': np.random.choice(['#BrandA', '#BrandB', '#BrandC', '#ProductX', '#ProductY', '#TopicZ'], size=data_size),
    'date': pd.date_range(start='2024-01-01', periods=data_size, freq='H')
})

# Visualize sentiment distribution across brands/topics

# Sentiment count by hashtag
sentiment_counts = social_media_data.groupby(['hashtag', 'sentiment']).size().unstack()

# Plotting sentiment distribution for each hashtag
plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='bar', stacked=True, color=['lightgreen', 'lightgray', 'lightcoral'], edgecolor='black')
plt.title('Sentiment Distribution by Hashtag', fontsize=15)
plt.ylabel('Number of Posts', fontsize=12)
plt.xlabel('Hashtag', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

# Plot sentiment over time (for trend analysis)
social_media_data['sentiment_score'] = social_media_data['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

# Aggregating sentiment scores by date
sentiment_trend = social_media_data.resample('D', on='date')['sentiment_score'].mean()

# Plotting sentiment trend over time
plt.figure(figsize=(10, 6))
sns.lineplot(x=sentiment_trend.index, y=sentiment_trend.values, color='blue', marker='o')
plt.title('Sentiment Trend Over Time', fontsize=15)
plt.ylabel('Average Sentiment Score', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()