# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython



# %%
import string

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy 
import praw
import streamlit as st



# %%

reddit = praw.Reddit(client_id='qQBQxY9R1zGrYJH9pACBOw', client_secret='MjrjCout_Omkh_3feZFq3UrzyLBicw', user_agent='Crypto')


# %%
st.title('Crypto Trader Assistant')
st.text('This tool gives traders and investors a sentiment about the cryptocurrency from r/Cryptocurrency.')

ucurr = st.text_input('Cryptocurrency', 'BTC')
st.write('Cryptocurrency', ucurr)


# %%
crypto = []
ml_subreddit = reddit.subreddit('Cryptocurrency')
for post in ml_subreddit.hot(limit=30000):
    crypto.append([post.title, post.score, post.id, post.subreddit, post.url, post.num_comments, post.selftext, post.created])
crypto = pd.DataFrame(crypto,columns=['title', 'score', 'id', 'subreddit', 'url', 'num_comments', 'body', 'created'])


# %%
crypto.dropna(subset=['body'], inplace=True)


# %%
crypto['original_body'] = crypto['body']


# %%
crypto


# %%
import spacy
nlp = spacy.blank('en')


import re

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stop_words(text):
    nlp.Defaults.stop_words |= {'nt','crypto', 'cryptocurrency', ' nt', 'nt '}
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

def lemmatize_words(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

remove_spaces = lambda x : re.sub('\\s+', ' ', x)

# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

remove_double_quotes = lambda x : x.replace('"', '')
remove_single_quotes = lambda x : x.replace('\'', '')
trim = lambda x : x.strip()

other_chars = ['*', '#', '&x200B', '[', ']', '; ',' ;' "&nbsp", "“","“","”", "x200b"]


def remove_other_chars(x: str):
    for char in other_chars:
        x = x.replace(char, '')
    
    return x


def lower_case_text(text):
    return text.lower()

funcs = [
    remove_urls, 
    remove_punctuation,
    lower_case_text,
    remove_stop_words, 
    remove_emoji, 
    remove_double_quotes, 
    remove_single_quotes,
    remove_other_chars,
    lemmatize_words,
    remove_spaces,
    trim]


# %%
for fun in funcs:
    crypto['body'] = crypto['body'].apply(fun)


# %%
crypto.reset_index(inplace=True)
crypto.drop(['index'], axis=1, inplace=True)


# %%
body_list = crypto.body.tolist()


# %%
def generate_ngrams(text, n_gram=2):
    token = [token for token in text.lower().split(' ') if token != '']
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]


# %%
from wordcloud import WordCloud

fig_wordcloud = WordCloud(stopwords=nlp.stopwords, background_color='lightgrey', 
                          colormap='viridis', width=800, height=600
                         ).generate(' '.join(body_list))

plt.figure(figsize=(10, 7), frameon=True)
plt.imshow(fig_wordcloud)
plt.axis('off')
plt.show()

st.pyplot(fig_wordcloud)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import linalg
from sklearn import decomposition
import fbpca

number_of_topics = 30
num_top_words = 20
vectorizer = TfidfVectorizer()


# %%
bigram_vectorizer = TfidfVectorizer(ngram_range=(2,2))


# %%
bigrams_vectors = bigram_vectorizer.fit_transform(body_list).todense()
bigrams_vocab = np.array(bigram_vectorizer.get_feature_names())

# %% [markdown]
# # LDA

# %%
docs = [generate_ngrams(body) for body in body_list]


# %%
docs


# %%
from gensim.corpora import Dictionary

dic = Dictionary(docs)


# %%
id = 0

for i in dic:
    if dic[i] == r'^crypto|$crypto':
        id = i


# %%
id


# %%
dic[1]


# %%
dic.dfs


# %%
corpus = [dic.doc2bow(doc) for doc in docs]


# %%
corpus

# %% [markdown]
# ## Training 
# 
# Now it's time to train our topic model. We do this with the following parameters:
# 
# - **corpus**: the bag-of-word representations of our documents
# - **id2token**: the mapping from indices to words
# - **num_topics**: the number of topics we want the model to identify
# - **chunksize**: the number of documents the model sees for every update
# - **passes**: the number of times we show the total corpus to the model during training
# - **random_state**: we use a seed to ensure reproducibility.

# %%
from gensim.models import LdaModel

model = LdaModel(corpus=corpus, id2word=dic, num_topics=number_of_topics, chunksize=2500, passes=5, random_state=1)


# %%
model.get_term_topics(130, minimum_probability=None)


# %%
# x=model.show_topics(num_topics=12, num_words=5,formatted=False)
# topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

# tid = null

# for topic,words in topics_words:
#     print(str(topic)+ ":"+ str(words))
#     for word in words
#         if word == 'btc'
#             tid = topic
#             break

# prin


# %%
x=model.show_topics()

twords={}
for topic,word in x:
    twords[topic]= re.split('[^A-Za-z ]+', '', word)
print(twords)


# %%
for (topic, words) in model.print_topics():
    print(topic+1, ":", words, '\n\n')


# %%
original_body_list = crypto.original_body.tolist()


# %%
for (text, doc) in zip(original_body_list[:9], docs[:9]):
    print('\033[1m' + 'Text: ' + '\033[0m', text)
    print('\033[1m' + 'Topics: ' + '\033[0m', [(topic+1, prob) for (topic, prob) in model[dic.doc2bow(doc)] if prob > 0.15])
    print('\n')

# %% [markdown]
# Now let's see what topic is going to assign to the post-**2179**, previously tested with the svd decomposition for the unigram term-document matrix.

# %%
print('\033[1m' + 'Text: ' + '\033[0m', original_body_list[20])
print('\033[1m' + 'Topic: ' + '\033[0m', [(topic+1, prob) for (topic, prob) in model[dic.doc2bow(docs[20])] if prob > 0.1])

# %% [markdown]
# the topics assigned by the LDA are not 100% spot-on, **topic 5** seems to be appropriate but **topic 9** seems to be out of place.
# 
# Let's see what kind of topic the LDA is going to choose for the post **280** tested with the bigram matrix decomposition.

# %%
print('\033[1m' + 'Text: ' + '\033[0m', original_body_list[30])
print('\033[1m' + 'Topic: ' + '\033[0m',[(topic+1, prob) for (topic, prob) in model[dic.doc2bow(docs[30])] if prob > 0.1])


# %%
from textblob import TextBlob
# create a function to get subjectivity
def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity

# create a function to get the polarity
def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity

# create two new columns called "Subjectivity" & "Polarity"
crypto['subjectivity'] = crypto['original_body'].apply(getSubjectivity)
crypto['polarity'] = crypto['original_body'].apply(getPolarity)


# %%
# create a function get the sentiment text
def getSentiment(score):
    if score < 0:
        return "negative"
    elif score == 0:
        return "neutral"
    else:
        return "positive"


# %%
# create a column to store the text sentiment
crypto['sentiment'] = crypto['polarity'].apply(getSentiment)


# %%
if __name__ == '__main__':
    main()
    import warnings
    warnings.filterwarnings('ignore')


