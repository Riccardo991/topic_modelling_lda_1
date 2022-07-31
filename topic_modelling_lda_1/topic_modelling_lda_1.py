
#Topic Modeling 1:
#In this notebook I develop a topic modeling. During the implementation I focused on the methods offered by sklearn pacage  to create the model.
# the dataset consists of approximately 12,000 NPR (National Public Radio) articles, obtained from their website www.npr.org

import pandas as pd 
import numpy as np 
import re, spacy, time 
from collections import Counter 
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nlp = spacy.load("en_core_web_sm")

# I clean up the text by eliminating punctuation, stopwords, verbs and pronouns.
# In this way I leave only the essential in the article, so as to simplify the recognition of the topic
def makeCorpus ( text ):            
    text = text.lower()
    text = text.replace("'", " ")
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')    
    text = re.sub( r'\S+@\S+', ' ', text)
    text = re.sub( '\w*\d\w*', ' ', text)
    text = re.sub( r'[^\w\s]', ' ', text)
    text = re.sub( r' +', ' ', text)
    if len(text) <= 4 :
        text = 'empty' 
    else:   
        doc = nlp(text)
        good = ''
        for tk in doc:        
            if tk.is_stop == False and tk.is_punct == False and tk.pos_ not in ['PRON', 'VERB']:
              good = good+tk.lemma_+' '
        text = good 
    text  = re.sub(r' +', ' ', text)                                   
    return   text 


# In getTopicWords I return the n most frequent words for each argument.
# With lda.components_ I take the list of words that the model has grouped into a topic, reorder them and select the n most frequent ones.
def getTopicWords ( lda, voc, n_words ):
    dc = {}
    for topic_idx, topic in enumerate( lda.components_):        
        top_features_id = topic.argsort()[:-n_words -1:-1]
        top_words = [ voc[i] for i in top_features_id ]
        w = 'Topic_'+str(topic_idx+1)
        dc[w] = top_words
    df = pd.DataFrame( dc )
    return df 

print("go")
df = pd.read_csv('Article.csv', sep=';')
print("df size ", df.shape)

t1 = time.time() 
df['Article'] = df['Article'].apply( str ) 
df['corpus'] = df['Article'].apply( makeCorpus ) 
df['dim'] = df['corpus'].apply(lambda x : len( x.split()))
print(" the mean of words in a clean text is ",df['dim'].mean())

t2 = time.time()
print(" the preprocessing time is ",int(t2-t1)) 


# I apply the CountVectorizer to the corpus of articles and filter by eliminating words with a frequency greater than 70% or that occur less than 10 times.
# So as to keep only the rare words specific to a topic.
cv = CountVectorizer(max_df=0.70,   min_df = 10 )
x_set = cv.fit_transform( df['corpus'].values )
x_set = x_set.toarray()
voc = cv.get_feature_names()
print(" words vocavolary size  ",len(voc))

# LDA model 
topic = 7
lda = LatentDirichletAllocation( n_components=topic, learning_method='online', random_state=23 )
# train 
lda.fit( x_set )
# evaluate 
l = lda.score( x_set)
p = lda.perplexity( x_set ) 
print(" Log Likelihood: %.4f, Perplexity %.4f " %(l, p))

# get the n most frequent words  for topic 
n_words = 15
topics_df = getTopicWords ( lda, voc, n_words)
print(" words for topics  df  ",topics_df.head() )

# I apply the LDA model on  the dataset to predict the topics of the articles
y_pred = lda.transform(x_set ).argmax( axis=1)
df['topic_label'] = y_pred 
df['topic_label'] = df['topic_label'].apply( lambda x: x+1)
print(" result distributions  ",Counter(df['topic_label'] )) 

df.to_csv('Article_with_topics_1.csv', sep=';', index=False)
topics_df.to_csv('words_for_topic_1.csv', sep=';', index=False)
print("end ")
