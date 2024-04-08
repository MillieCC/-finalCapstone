# Captone Project - Cheung Wing Yan

#To implement a sentiment analysis model with spaCy and load spaCy model
import numpy as np
import pandas as pd
import spacy
from textblob import TextBlob

# To Load SpaCy model
nlp = spacy.load('en_core_web_sm')

# Load the dataset
amazon = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')

# Clean text data. Remove missing values
clean_data = amazon.dropna(subset=['reviews.text'])

# Display the cleaned data
print(clean_data.head())

# Define function for text preprocessing.
def preprocess(text):
    doc = nlp(text.lower().strip())
    processed = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(processed)

# Apply text preprocessing to 'reviews.text' column in csv file
clean_data['processed_text'] = clean_data['reviews.text'].apply(preprocess)

# Define function for performing sentiment analysis using TextBlob with clean data
def analyze_sentiment(text):
    analysis = TextBlob(text)   
    sentiment_score = analysis.sentiment.polarity
    return 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral'

# Apply sentiment analysis to preprocessed text
clean_data['sentiment'] = clean_data['processed_text'].apply(analyze_sentiment)

# Display the results
print(clean_data[['reviews.text', 'processed_text', 'sentiment']].head())

# Test out the model on sample product review

review = "These Amazon batteries did the job although I gave 4star only because I had a few I would say a hand full of batteries that were not as strong or were pretty weak but out of a box of 48 batteries, I will definitely buy again for this priceIm pretty well satisfied.Thank you!"
preprocessed_review = preprocess(review)
sentiment = analyze_sentiment(preprocessed_review)
print("Sentiment:", sentiment)
