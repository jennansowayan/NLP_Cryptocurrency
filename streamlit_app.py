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

ucurr = st.text_input('Enter a word related to the Cryptocurrency world:', 'e.g. btc')
st.balloons()


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
import spacy
nlp = spacy.blank('en')


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_stop_words(text):
    doc = nlp(text)
    nlp.Defaults.stop_words |= {'nt','crypto', 'cryptocurrency', ' nt', 'nt '}
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
    lower_case_text,
    remove_punctuation,
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
from gensim.corpora import Dictionary

dic = Dictionary(docs)


# %%
corpus = [dic.doc2bow(doc) for doc in docs]


# %%
from gensim.models import LdaModel

model = LdaModel(corpus=corpus, id2word=dic, num_topics=number_of_topics, chunksize=2500, passes=5, random_state=1)


# %%
x = model.show_topics(num_topics=30, num_words=20,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

tid = np.nan
flag = 0
for topic,words in topics_words:
    print(str(topic)+ ":"+ str(words))
    for word in words:
        if len(re.findall('('+'btc'+')', word)):
            tid = topic
            flag = 1
            break
    if flag:
        break
        
      


# %%
if np.isnan(tid):
     st.error('Topic not found, please revise the spelling or enter another topic.')
else:
     ilist = []
     for doc in range(len(docs)):
          prob = model.get_document_topics(dic.doc2bow(docs[doc]),minimum_probability=0)
          if prob[tid][1] > 0.15:
               ilist.append(doc)
               
print(ilist)


# %%
from textblob import TextBlob
# create a function to get subjectivity
def getSubjectivity(twt):
    return TextBlob(twt).sentiment.subjectivity

# create a function to get the polarity
def getPolarity(twt):
    return TextBlob(twt).sentiment.polarity

# create two new columns called "Subjectivity" & "Polarity"
crypto['subjectivity'] = crypto['original_body'][ilist].apply(getSubjectivity)
crypto['polarity'] = crypto['original_body'][ilist].apply(getPolarity)


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
def getpercentage(df):
    p = df.sentiment.loc[ df.sentiment == 'positive'].count()
    neg = df.sentiment.loc[ df.sentiment == 'negative'].count()
    n = df.sentiment.loc[ df.sentiment == 'neutral'].count()
    total = p + neg + n
    p1, p2, p3 = p/total, n/total, neg/total
    return [p1*100, p2*100, p3*100] 


# %%
df=crypto.loc[ilist]
values =getpercentage(df)
labels = ['Positive', 'Neutral', 'Negative']


# %%
values


# %%
colors = ['blue','grey','red']


# %%
import plotly.graph_objects as go
import plotly.express as px
fig6 = go.Figure(data = go.Pie(values = values, 
                               labels = labels, hole = 0.8,
                               marker_colors = colors ))
fig6.update_traces(hoverinfo='label+percent',
                   textinfo='percent', textfont_size=20)
fig6.add_annotation(x= 0.5, y = 0.5,
                    text = 'Sentiments',
                    font = dict(size=20,family='Verdana', 
                                color='black'),
                    showarrow = False)
fig6.show()


# %%
st.pyplot(fig6)


# %%
if __name__ == '__main__':
    main()
    import warnings
    warnings.filterwarnings('ignore')


