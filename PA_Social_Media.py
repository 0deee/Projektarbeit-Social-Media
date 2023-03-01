# data processing
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.corpus import wordnet

# ML model related
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#  plotting graphs
import matplotlib.pyplot as plt

# misc
import time # measure time
from pprint import pprint # pretty print for testing purposes
import logging # console logging for debugging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#load list dependencies for preprocessing
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

#setup stopwords
stop_words = set(stopwords.words('english'))

# load csv files into dataframes
df_tweet = pd.read_csv('data/Tweet.csv')
df_company = pd.read_csv('data/Company.csv')
df_company_tweet = pd.read_csv('data/Company_Tweet.csv')

# merge Company.csv and Company_Tweet.csv into temporary dataframe
temp_company_tweet = pd.merge(df_company_tweet, df_company, on='ticker_symbol')

#merge temp dataframe with df_tweet for final usable dataframe
tweets_main = pd.merge(temp_company_tweet, df_tweet, on='tweet_id')

#convert post_date into datetime and remove time
tweets_main['post_date'] = pd.to_datetime(tweets_main['post_date'], unit='s').apply(lambda date: date.date())

#filter dataframe by 2019 data
tweets_main = tweets_main[tweets_main['post_date'].dt.year==2019]

#filter dataframe by TSLA ticker (removing TSLA)
tweets_main = tweets_main[tweets_main['ticker_symbol'] != 'TSLA']

#remove all duplicate tweets
tweets_main = tweets_main.drop_duplicates(subset='tweet_id')


#data preprocessing
# functions for topic detection preprocessing
def removeSpecialChars(tweet_text):
    tweet_text =re.sub(r'\bRT\b', '', tweet_text) #remove RT (Retweet)
    tweet_text = re.sub(r'http\S+', '', tweet_text) #remove URLS
    tweet_text = re.sub(r'[$]\w+', '', tweet_text) # remove stock ticker e.g. $AAPL
    tweet_text = re.sub(r'^#\s|#\'|\s# ', 'number', tweet_text) #replace # with 'number'
    tweet_text = re.sub(r'[0-9]', '', tweet_text) #remove numbers
    tweet_text = re.sub(r'[^\w\s\']', '', tweet_text)
    return tweet_text
    
def tokenizeText(tweet_text):
    tokens = word_tokenize(tweet_text) # tokenize each word
    return tokens

def posTagging(tokens):
    pos_tokens = nltk.pos_tag(tokens) # set POS tag for each token in a tweet
    return pos_tokens

def removeStopwords(tokens):
    filtered_tokens = []
    #filtered_tokens = [token for token in tokens if not token.lower() in stop_words]
    tagged_tokens = posTagging(tokens)
    #remove all word types that are not nouns and verbs and N and V that are on stopword list 
    for word, tag in tagged_tokens:
        if tag.startswith('N') or tag.startswith('V'):
            if word.lower() in stop_words:
                continue
            else:
                filtered_tokens.append(word)
        else:
            if word.lower() in stop_words:
                continue
    return filtered_tokens

def lemmatizeText(tokens):
    lemmatizer = WordNetLemmatizer()
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
 

def preprocessDataTD(dataset):
    dataset['body'] = dataset['body'].apply(removeSpecialChars)
    dataset['tokens'] = dataset['body'].apply(tokenizeText)
    dataset['pos_tagged_tokens'] = dataset['tokens'].apply(posTagging)
    dataset['lemmatized_tokens'] = dataset['pos_tagged_tokens'].apply(lemmatizeText)
    dataset['filtered_tokens'] = dataset['lemmatized_tokens'].apply(removeStopwords)
    dataset['filtered_tokens'] = dataset['filtered_tokens'].apply(lowercasingText)
    return dataset

def preprocessDataSA(tweet_text):
    tweet_text =re.sub(r'\bRT\b', '', tweet_text) #remove RT (Retweet)
    tweet_text = re.sub(r'http\S+', '', tweet_text) #remove URLS
    tweet_text = re.sub(r'@\w+\s?', '', tweet_text) #remove @username
    tweet_text = re.sub(r'#\w+', '', tweet_text) #remove hashtags
    tweet_text = re.sub(r'^#|#\'|\s#', 'number', tweet_text) #replace # with 'number'
    return tweet_text

# functions for model building
def runLDAModel(dataset):
    # create dictionary for LDA
    td_prepped = dataset['filtered_tokens'].copy()
    dictionary = Dictionary(td_prepped)
    #dictionary.filter_extremes(no_below=20, no_above=0.5) # filter extreme values

    # create corpus
    corpus = [dictionary.doc2bow(tweets) for tweets in td_prepped]

    # train optimized model
    lda_optimized = LdaModel(corpus=corpus,
                     id2word=dictionary,
                     num_topics=12,
                     alpha='symmetric',
                     eta='symmetric',
                     )
    return lda_optimized


def replaceSentScores(score):
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def get_sentiment_scores(text):
    scores = SentimentIntensityAnalyzer().polarity_scores(text) # check 
    return scores

def runVADERModel(dataset):
    sentiment_scores = []
    for text in dataset['body']:
        sentiment_scores.append(get_sentiment_scores(text))  # for each tweet get sentiment score
    
    sentiment_df = pd.DataFrame(sentiment_scores) # sentiment score to dataframe
    dataset = dataset.reset_index(drop=True) #reset index on tweet dataset for concat
    sentiment_concat = pd.concat([dataset, sentiment_df], axis=1) # concat tweets and sentiment results
    sentiment_concat['sentiment_score'] = sentiment_concat['compound'].apply(replaceSentScores) # calculate sentiment classifications 
    return sentiment_concat

###########################################################################################

# DP topic detection
def runTopicDetection():
    # drop all columns but tweet 'body'
    td_cleaned = tweets_main.drop(['tweet_id', 'ticker_symbol','company_name','writer',
                              'post_date','comment_num','retweet_num','like_num'], axis=1)
    preprocessDataTD(td_cleaned) # preprocess data
    lda_optimized = runLDAModel(td_cleaned) # run LDA
    pprint('Resulting Topics of LDA Topic Modeling:')
    pprint(lda_optimized.print_topics())

    # assign topics to each tweet
    doc_topics = lda_optimized[corpus] # get topic distribution

    topic_assignments = [max(topics, key=lambda x: x[1])[0] for topics in doc_topics] # assign document by highest probability
    td_assigned = td_cleaned.copy()
    td_assigned['topic_no'] = topic_assignments


    topic_mapping = {0: 'MS Azure Growth', 1: 'Security Alerts & Vulnerabilitiy Detection',
                2: 'Stock Market News', 3: 'Crypto & Blockchain Technology', 4: 'Amazon Stock Market Activity',
                5: 'Daily Stock Market Trading', 6: 'Cloud Technology News', 7: 'Financial Analysis',
                8: 'Microsoft Market Performance', 
                9: 'Finance Advertising', 10: 'Trading Strategies', 
                11: 'Microsoft Today'}
    
    

# DP sentiment analysis
def runSentimentAnalysis():
    # drop all columns but tweet 'body'
    sa_cleaned = tweets_main.drop(['tweet_id', 'ticker_symbol','company_name','writer',
                              'post_date','comment_num','retweet_num','like_num'], axis=1)
    sa_cleaned['body'] = sa_cleaned['body'].apply(preprocessDataSA) # preprocess data 
    sentiment_concat = runVADERModel(sa_cleaned) # run VADER model
    return sentiment_concat


#### dominant topic per tweet
#### put together td and sa
### print out bar chart and topics

def runTextAnalysis():
    return