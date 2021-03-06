#!/usr/bin/env python
# coding: utf-8

# # The goal of this project was to use Latent Semantic Analysis to determine the dominant words and concepts associated with a dataset of described UFO sightings and create topic clusters.

# In[1]:


###Importing Packages
import pandas as pd
import numpy as np 
import os


# In[2]:


###Setting Working Directory
import os
path="/Users/benroberts/Downloads/MSA-Fall1"
os.chdir(path)
os.getcwd()


# In[3]:


###Loading Dataset as Dataframe
ufo = pd.read_csv("scrubbed.csv")


# In[8]:


### Subsetting Comments
ufo_com=ufo['comments']


# In[143]:


###Creating list of strings for Processing
ufo_l=list(ufo_com)
type(ufo_l)
for i in range(len(ufo_l)):
    ufo_l[i]=str(ufo_l[i])
#ufo_l


# In[10]:


###Importing Text Packages
import nltk
import re
import string


# In[11]:


### Remove punctuation, capitals, then tokenize documents

punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
term_vec = [ ]

for d in ufo_l:
    d = d.lower()
    d = punc.sub( '', d )
    term_vec.append( nltk.word_tokenize( d ) )


# In[12]:


### Remove stop words from term vectors

stop_words = nltk.corpus.stopwords.words( 'english' )

for i in range( 0, len( term_vec ) ):
    term_list = [ ]

    for term in term_vec[ i ]:
        if term not in stop_words:
            term_list.append( term )

    term_vec[ i ] = term_list


# In[13]:


### Porter stem remaining terms

porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )


# In[14]:


###Detokenize term vector to Return New List of Comments
detokenized_doc = []
for i in range(len(term_vec)):
    t = ' '.join(term_vec[i])
    detokenized_doc.append(t)

ufo_new = detokenized_doc


# In[15]:


### TF-IDF vectorize documents w/sklearn, remove English stop words

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words

vect = TfidfVectorizer( stop_words='english', max_features=10000 )
tfidf_matrix = vect.fit_transform( ufo_new )
tf_idf_matrix = tfidf_matrix.todense()


# In[16]:


### Create list of remaining unique terms (10,000)
terms = vect.get_feature_names()


# In[17]:


###TFIDF DataFrame
tfidf=pd.DataFrame(tf_idf_matrix, columns=terms)


# In[29]:


###Latent Semantic Analysis with SVD on TF-IDF Matrix (15 concepts)

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=500, random_state=42)

document_topics=svd_model.fit_transform(tfidf_matrix)


# In[247]:


###Printing Out Topics/Concepts and their most Heavily Weighted Terms
term=[]
weight=[]
yeah=abs(svd_model.components_)
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
    for j in range(len(sorted_terms)):
        temp.append(sorted_terms[j][0])
    print("Topic "+str(i)+": ")

    for t in sorted_terms:
        print(t[1], t[0])
        term.append(t[0])
        weight.append(t[1])
        print(" ")


# In[ ]:





# In[178]:


###Pulling Out Terms and Weights For Each Topic

weight
term
#term[4]='orange'
#term[9]='fly'




# In[ ]:





# In[176]:


###Dictionary of Terms and Weights for Topic 0
dicts = {}
for t in range(len(term)):
    dicts[term[t]] = weight[t]
print(dicts)


# In[179]:


###WordCloud of Highest Weighted Words for Topic 0
import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

wc = WordCloud(
    background_color="white",
    
    max_words=2000,
    width = 1024,
    height = 720,
    colormap=matplotlib.cm.inferno
)

# Generate the cloud

wc.generate_from_frequencies(dicts)

# Save the could to a file

wc.to_file("topic_all.png")


# In[32]:


###Term-Concept Matrix
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
concept_term_mat=pd.DataFrame(np.round(svd_model.components_,4),index = [['T'+str(i) for i in range(1,11)]],columns =
terms)


# In[33]:


###Document-Concept Matrix
doc_concept=pd.DataFrame(np.round(document_topics,3), columns=['T'+str(i) for i in range(1,11)])
doc_concept.head()


# In[226]:


#Sorting Document-Concept Matrix by highest weighted terms in each topic
doc_concept.sort_values(by='T5',ascending=False).head()


# In[242]:


###TOP 5 COMMENTS FOR EACH CONCEPT
columns=list(doc_concept)
x=[doc_concept[column].nlargest(5).index.values for column in doc_concept]
for i in range(0,10):
    for j in range(0,5):
        print('Topic'+ ' ' + str(i+1)+ ' ' +ufo_l[x[i][j]])
###For Each Topic, Printing Comments with Highest Scores


# In[282]:


Sigma_matrix = np.diag(svd_model.singular_values_)

#Topics 1, 3, 4 and 10. July 4th as most frequent date


# In[292]:


###Determining Highest Scored Topic for Each Document
topic_cluster=doc_concept.idxmax(axis=1)


# In[288]:


x=np.dot(doc_concept.values,Sigma_matrix)
doc_weight=pd.DataFrame(x, columns=columns)
doc_weight.head()


# In[44]:


doc_concept.head()


# In[293]:


###Appending Dominant Topic to Each Document in DataFrame
ufo['topic']=topic_cluster


# In[47]:


ufo.head()


# In[56]:


###Reading DataFrame with Dominant Topic Appended to Each Document to CSV
ufo.to_csv(r'/Users/benroberts/Downloads/MSA-Fall1/Python/ufo_topic.csv',index=None)


# In[291]:


ufo['topic'].value_counts()


# In[296]:


###Read in and view New Dataset
x=pd.read_csv('df_new.csv')


# In[298]:


x.head()


# In[ ]:




