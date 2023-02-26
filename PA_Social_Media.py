import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim.corpora import Dictionary
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#load list dependencies
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

#setup stopwords
stop_words = set(stopwords.words('english'))

#import all csv files
df_tweet = pd.read_csv("data/Company_Tweet.csv")
df_company = pd.read_csv("data/Company.csv")
df_company_tweet = pd.read_csv("data/Company_Tweet.csv")

#merge df_company_tweet and df_company in temp dataframe
temp_company_tweet = pd.merge(df_company_tweet, df_company, on="ticker_symbol")

#merge temp dataframe with df_tweet for final usable dataframe
tweets_main = pd.merge(temp_company_tweet, df_tweet, on="tweet_id")

#convert post_date into datetime and remove time
tweets_main['post_date'] = pd.to_datetime(tweets_main['post_date'], unit='s').apply(lambda date: date.date())

#filter dataframe by 2019 data
tweets_main = tweets_main[tweets_main["post_date"].dt.year==2019]

#filter dataframe by TSLA ticker (removing TSLA)
tweets_main = tweets_main[tweets_main["ticker_symbol"] != "TSLA"]

#remove all duplicate tweets
tweets_main = tweets_main.drop_duplicates(subset="tweet_id")

#########################################################################################################################
#data preprocessing
def removeSpecialChars(tweet_text):
    tweet_text =re.sub(r'\bRT\b', '', tweet_text) #remove RT (Retweet)
    tweet_text = re.sub(r'http\S+', '', tweet_text) #remove URLS
    tweet_text = re.sub(r'@\w+\s?', '', tweet_text) #remove @username
    #tweet_text = re.sub(r'(?<!\S)@', 'at', tweet_text) #@ sign replace = \s@ = @ alone \b@ = word@ \(@ 
    tweet_text = re.sub(r'#\w+', '', tweet_text) #remove hashtags
    tweet_text = re.sub(r'^#|#\'|\s#', 'number', tweet_text) #replace # with "number"
    tweet_text = re.sub(r'[0-9]', '', tweet_text) #remove numbers
    tweet_text = re.sub(r'[^\w\s\']', '', tweet_text)
    return tweet_text
    
def tokenizeText(tweet_text):
    tokens = word_tokenize(tweet_text)
    return tokens

def posTagging(tokens):
    pos_tokens = nltk.pos_tag(tokens)
    return pos_tokens

def removeStopwords(tokens):
    #stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    filtered_tokens = [token for token in tokens if not token.lower() in stop_words]
    #tagged_tokens = posTagging(tokens)
    #remove adjetives, adverbs and modals
#     for word, tag in tagged_tokens:
#         if tag.startswith('J') or tag.startswith('R') or tag.startswith('M'):
#             continue
#         if word.lower() in stop_words:
#             continue
#         filtered_tokens.append(word)
    return filtered_tokens

def lemmatizeText(tokens):
    lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_tokens = []
    for token, pos in tokens:
        if pos.startswith('V'):
            lemma = lemmatizer.lemmatize(token, pos='v')
        elif pos.startswith('J'):
            lemma = lemmatizer.lemmatize(token, pos='a')
        elif pos.startswith('R'):
            lemma = lemmatizer.lemmatize(token, pos='r')
        else:
            lemma = lemmatizer.lemmatize(token)
        lemmatized_tokens.append(lemma)
    return lemmatized_tokens

def lowercasingText(tweet_text):
    lowercased_text = [tweet.lower() for tweet in tweet_text]
    return lowercased_text #lowercasing

 #########################################################################################################################   

def preprocessDataTD(dataset):
    dataset["body"] = dataset["body"].apply(removeSpecialChars)
    dataset['tokens'] = dataset['body'].apply(tokenizeText)
    dataset['pos_tagged_tokens'] = dataset['tokens'].apply(posTagging)
    dataset['lemmatized_tokens'] = dataset['pos_tagged_tokens'].apply(lemmatizeText)
    dataset['filtered_tokens'] = dataset['lemmatized_tokens'].apply(removeStopwords)
    dataset['filtered_tokens'] = dataset['filtered_tokens'].apply(lowercasingText)

def preprocessDataSA(tweet_text):
    tweet_text =re.sub(r'\bRT\b', '', tweet_text) #remove RT (Retweet)
    tweet_text = re.sub(r'http\S+', '', tweet_text) #remove URLS
    tweet_text = re.sub(r'@\w+\s?', '', tweet_text) #remove @username
    #tweet_text = re.sub(r'(?<!\S)@', 'at', tweet_text) #@ sign replace = \s@ = @ alone \b@ = word@ \(@ 
    tweet_text = re.sub(r'#\w+', '', tweet_text) #remove hashtags
    tweet_text = re.sub(r'^#|#\'|\s#', 'number', tweet_text) #replace # with "number"

    return tweet_text

#########################################################################################################################

def runLDAModel(dataset):
    return dataset


def replaceSentScores(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_sentiment_scores(text):
    scores = SentimentIntensityAnalyzer().polarity_scores(text) # check 
    return scores

def runVADERModel(dataset):
    sentiment_scores = []
    for text in dataset["body"]: # for each row in tweet set
        sentiment_scores.append(get_sentiment_scores(text))
    
    sentiment_df = pd.DataFrame(sentiment_scores) # transform to dataframe
    dataset = dataset.reset_index(drop=True) #reset index on tweet dataset for concat
    sentiment_concat = pd.concat([dataset, sentiment_df], axis=1) # concat tweets and sentiment results
    sentiment_concat["sentiment_score"] = sentiment_concat["compound"].apply(replaceSentScores)   

    return sentiment_concat

#########################################################################################################################
# DP topic detection
def runTopicDetection():
    # drop all columns but tweet "body"
    td_cleaned = tweets_main.drop(["tweet_id", "ticker_symbol","company_name","writer",
                              "post_date","comment_num","retweet_num","like_num"], axis=1)
    preprocessDataTD(td_cleaned) # preprocess data
    runLDAModel(td_cleaned) # run LDA
    

# DP sentiment analysis
def runSentimentAnalysis():
    # drop all columns but tweet "body"
    sa_cleaned = tweets_main.drop(["tweet_id", "ticker_symbol","company_name","writer",
                              "post_date","comment_num","retweet_num","like_num"], axis=1)
    sa_cleaned["body"] = sa_cleaned["body"].apply(preprocessDataSA) # preprocess data 
    runVADERModel(sa_cleaned) # run VADER model