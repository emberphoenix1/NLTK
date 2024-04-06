from nltk.sentiment import SentimentIntensityAnalyzer

# Create a SentimentIntensityAnalyzer instance
sia = SentimentIntensityAnalyzer()

# Analyze sentiment for a sample text
text = "woof woof arf"

# Get the sentiment scores dictionary
sentiment_scores = sia.polarity_scores(text)

# Print out the compound score directly
print("Sentiment Score for the text:", text)
print("Compound Score:", sentiment_scores['compound'])
