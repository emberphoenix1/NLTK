import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')

def analyze_sentiment_and_generate_response(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        return "I'm glad to hear that!"
    elif compound_score <= -0.05:
        return "I'm sorry to hear that. Is there anything I can do to help?"
    else:
        return "Thanks for sharing. Let me know if you have any other questions."

def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]
    
    return lemmatized_tokens

def chat():
    print("Hello! I am a chat bot. How are you doing today?")
    while True:
        message = input("> ").lower()
        if message.lower() == 'bye':
            print("Goodbye! Have a great day.")
            break
        else:
            processed_message = preprocess_text(message)
            # Join the list of tokens into a single string
            processed_message_str = ' '.join(processed_message)
            response = analyze_sentiment_and_generate_response(processed_message_str)
            print(response)

if __name__ == "__main__":
    chat()
