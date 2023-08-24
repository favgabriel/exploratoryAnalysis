#import tweepy
#from bertopic import BERTopic
import praw
import csv
import pandas as pd
import numpy as np
import nltk
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# To import spacy for lemmatization
import spacy
import seaborn as sns
from textblob import TextBlob
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import collections
#nltk.download('stopwords')
#nltk.download('omw-1.4')
#nltk.download('wordnet')

wn = nltk.WordNetLemmatizer()
# key = "KbQOv8ktxuciZKM0I2IqbgU1g"
# secret = "1rNnav1K7wbzSyFHjHQBKvrZzMJdGRdXG7NVG5FUOY3BwoF2tW"
# token = "1501951522107629569-rOOKmy9orsF50TiSLK1PACivMTgS18"
# token_secret = "3HO00yKTxE8ArmXBF9SfBxa0oSUm82uYDMevbo57b0a1w"

# auth = tweepy.OAuthHandler(key,secret)
# auth.set_access_token(token,token_secret)
#
# api = tweepy.API(auth)
#
query = "TeslaLounge"
#
# tweets = tweepy.Cursor(api.search_tweets, query, lang="en").items(500)
#
# with open('tesla_tweet.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     for tweet in tweets:
#         writer.writerow(tweet.text)
#
# data = pd.read_csv('tesla_tweet.csv')

clientid="KpiS6WBzoHYU8LRPfIsCyg"
clientsecret= "0IJzKQNgWbqSIi6NpkQBF641L7NR4g"
# reddit = praw.Reddit(client_id=clientid,client_secret=clientsecret,password="arolines1",user_agent="praw 1",username="Space_avenger")
# subreddit = reddit.subreddit(query)
# titles=[]
# scores=[]
# ids=[]
#
# for submission in subreddit.top(limit=500):
#     titles.append(submission.title)
#     scores.append(submission.score)
#     ids.append(submission.id)
#
# df = pd.DataFrame()
#
# df['id']=ids
# df['titles']=titles
# df['score']=scores
#
# df.to_csv('social.csv',sep=',')

df = pd.read_csv('social.csv')

# print(df.head())
# print(df.describe())


'''topic modelling'''
# def content_to_words(tweet_content):
#     for tweet in tweet_content:
#         yield(gensim.utils.simple_preprocess(str(tweet), deacc = True))
#
# tweet_words = list(content_to_words(df['titles']))
#
# bigram = gensim.models.Phrases(tweet_words, min_count = 5, threshold = 100)
# trigram = gensim.models.Phrases(bigram[tweet_words], threshold = 100)
#
# bigram_mod = gensim.models.phrases.Phraser(bigram)
# trigram_mod = gensim.models.phrases.Phraser(trigram)
#
# for word in trigram_mod[bigram_mod[tweet_words[10:20]]]:
#     print(word)

# applying stop words and lem
stemmer = SnowballStemmer('english')
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_tweets = df['titles'].map(preprocess)

# for word in list(tweet_words):  # iterating on a copy since removing will mess things up
#     if word in list(STOPWORDS):
#         tweet_words.remove(word)
#
# # Define functions for stopwords, bigrams, trigrams and lemmatization
# def remove_STOP_WORDS(texts):
#     return [[word for word in simple_preprocess(str(doc)) if word not in list(STOPWORDS)] for doc in texts]
#
# def make_bigrams(texts):
#     return [bigram_mod[doc] for doc in texts]
#
# def make_trigrams(texts):
#     return [trigram_mod[bigram_mod[doc]] for doc in texts]
#
# def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
#     """https://spacy.io/api/annotation"""
#     texts_out = []
#     for sent in texts:
#         doc = nlp(" ".join(sent))
#         texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
#     return texts_out
#
#
# # To remove stopwords
# tweet_words_nostopword = remove_STOP_WORDS(tweet_words)
#
# # To form bigrams
# tweet_words_bigrams = make_bigrams(tweet_words_nostopword)
#
# nlp = spacy.load('en_core_web_sm', disable = ['parser', 'ner'])
#
# # To lematize while keeping only nouns, adjectives, verbs and adverbs
# tweet_lemmatized = lemmatization(tweet_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#
# print(tweet_lemmatized[10:20])
#
# id2word = corpora.Dictionary(tweet_lemmatized)
#
# texts = tweet_lemmatized
#
# corpus = [id2word.doc2bow(text) for text in texts]
#
# print(corpus[10:20])

#print(processed_tweets[:10])

# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#
#     coherence_values = []
#     model_list = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, random_state=100, update_every=1, chunksize=2000, passes=20, alpha='auto', per_word_topics=True)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())
#
#     return model_list, coherence_values
#
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=tweet_lemmatized, start=2, limit=40, step=6)
#
# # To print the coherence scores to determine the highest coherence value
# for m, cv in zip(model_list, coherence_values):
#     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=10, random_state=100, update_every=1, chunksize=2000, passes=20, alpha='auto', per_word_topics=True)

lexicon = gensim.corpora.Dictionary(processed_tweets)

# count = 0
# for k, v in lexicon.iteritems():
#     print(k, v)
#     count += 1
#     if count > 10:
#         break

# print(lexicon.token2id)

# words= {}
# for n in lexicon.token2id:
#     words[n] += 1
word_count = collections.Counter(lexicon.token2id)

lst = word_count.most_common(10)

dc = pd.DataFrame(lst, columns=['word','count'])

dc.plot.bar(x='word',y='count')
plt.show()

bow_corpus = [lexicon.doc2bow(doc) for doc in processed_tweets]
bow_corpus[86]

#lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=lexicon, passes=2, workers=2)
lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, id2word=lexicon, num_topics=10, random_state=100, update_every=1, chunksize=2000, passes=20, alpha='auto', per_word_topics=True)

#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, lexicon)
vis

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


#sentiment analysis

#Analyse the subjectivity
def subjectivity(text):
  return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
   return  TextBlob(text).sentiment.polarity

# Create two new columns 'Subjectivity' & 'Polarity'
df["Subjectivity"] = df["titles"].apply(subjectivity)
df['Polarity'] = df['titles'].apply(getPolarity)

def getAnalysis(score):
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'

df['Analysis'] = df['Polarity'].apply(getAnalysis)
df.to_csv('social_model_v2', index = False)

def getAnalysissub(score):
    if score < 0:
      return 'Negative'
    elif score == 0:
      return 'Neutral'
    else:
      return 'Positive'

df['Sub_Analysis'] = df['Subjectivity'].apply(getAnalysis)

#print(df['Sub_Analysis'].value_counts())
# plt.subplot(3,2,1)
# df['Analysis'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Polarity')
#plt.show()

# plt.subplot(3,2,2)
# df['Sub_Analysis'].value_counts(normalize=True).plot.bar(figsize=(20,10), title='Subjectivity')
#plt.show()

#sns.countplot(df['Analysis'])
#sns.countplot(df['Sub_Analysis'])

# plt.subplot(3,1,3)
# for i in range(0, df.shape[0]):
#    plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue')
#
#
# plt.title('Sentiment Analysis')
# plt.xlabel('Polarity')
# plt.ylabel('Subjectivity')
# plt.show()