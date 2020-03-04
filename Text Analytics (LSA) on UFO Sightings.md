
# The goal of this project was to use Latent Semantic Analysis to determine the dominant words and concepts associated with a dataset of described UFO sightings and create topic clusters.

### Data Prep


```python
###Importing Packages
import pandas as pd
import numpy as np 
import os
```


```python
###Setting Working Directory
import os
path="/Users/benroberts/Downloads/MSA-Fall1"
os.chdir(path)
os.getcwd()
```




    '/Users/benroberts/Downloads/MSA-Fall1'




```python
###Loading Dataset as Dataframe
ufo = pd.read_csv("scrubbed.csv")
```

    /Users/benroberts/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3049: DtypeWarning: Columns (5,9) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
### Subsetting Comments
ufo_com=ufo['comments']
```


```python
###Creating list of strings for Processing
ufo_l=list(ufo_com)
type(ufo_l)
for i in range(len(ufo_l)):
    ufo_l[i]=str(ufo_l[i])
#ufo_l
```


```python
###Importing Text Packages
import nltk
import re
import string
```


```python
### Remove punctuation, capitals, then tokenize documents

punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
term_vec = [ ]

for d in ufo_l:
    d = d.lower()
    d = punc.sub( '', d )
    term_vec.append( nltk.word_tokenize( d ) )
```


```python
### Remove stop words from term vectors

stop_words = nltk.corpus.stopwords.words( 'english' )

for i in range( 0, len( term_vec ) ):
    term_list = [ ]

    for term in term_vec[ i ]:
        if term not in stop_words:
            term_list.append( term )

    term_vec[ i ] = term_list
```


```python
### Porter stem remaining terms

porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )
```


```python
###Detokenize term vector to Return New List of Comments

detokenized_doc = []
for i in range(len(term_vec)):
    t = ' '.join(term_vec[i])
    detokenized_doc.append(t)

ufo_new = detokenized_doc
```


```python
### TF-IDF vectorize documents w/sklearn, remove English stop words

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words

vect = TfidfVectorizer( stop_words='english', max_features=10000 )
tfidf_matrix = vect.fit_transform( ufo_new )
tf_idf_matrix = tfidf_matrix.todense()
```


```python
### Create list of remaining unique terms (10,000)
terms = vect.get_feature_names()
```


```python
###TFIDF DataFrame
tfidf=pd.DataFrame(tf_idf_matrix, columns=terms)
```

### Latent Semantic Analysis 


```python
###Latent Semantic Analysis with SVD on TF-IDF Matrix (15 concepts)

from sklearn.decomposition import TruncatedSVD

# SVD represent documents and terms in vectors 
svd_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=500, random_state=42)

document_topics=svd_model.fit_transform(tfidf_matrix)
```


```python
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
```

    Topic 0: 
    0.5385734656359412 light
     
    0.3617031187922644 sky
     
    0.3159090815223304 bright
     
    0.25177545629379994 object
     
    0.20269082186896123 orang
     
    0.17267533619121195 red
     
    0.17001387110671765 white
     
    0.15629877953607885 shape
     
    0.11044498769831661 flash
     
    0.10807283075836578 fli
     
    Topic 1: 
    0.6076086111310213 object
     
    0.35157414932409503 shape
     
    0.18570067896005463 fli
     
    0.12140584651786682 ufo
     
    0.11306173406386788 craft
     
    0.09936046875492628 like
     
    0.08868659931660017 seen
     
    0.08804597887504013 triangl
     
    0.0819008302147879 triangular
     
    0.08148463192547972 saw
     
    Topic 2: 
    0.6575063468490188 sky
     
    0.3832803534638034 orang
     
    0.19082793594180836 firebal
     
    0.17800082334585365 night
     
    0.090203776630442 glow
     
    0.08951666773271823 ball
     
    0.07980550447369046 orb
     
    0.04691119850918314 object
     
    0.043691598353301934 sphere
     
    0.03075574630383237 strang
     
    Topic 3: 
    0.6891570751014322 orang
     
    0.1526174809404168 ball
     
    0.150473004785703 glow
     
    0.14727023210372028 craft
     
    0.14269762330943025 orb
     
    0.14136055989963062 triangl
     
    0.13193951481309563 shape
     
    0.1310940307722521 fli
     
    0.11597383156385913 format
     
    0.07456181613951442 sphere
     
    Topic 4: 
    0.5222938776481061 bright
     
    0.35579737144431184 object
     
    0.1532596770471333 orang
     
    0.063979813658856 fast
     
    0.06237018950887306 white
     
    0.06080654604680967 disappear
     
    0.04664898864438141 travel
     
    0.04461121584398084 speed
     
    0.03955264519978765 round
     
    0.03345916918916808 glow
     
    Topic 5: 
    0.4223321787570734 ufo
     
    0.3605209129053968 bright
     
    0.27834297124471286 sight
     
    0.2758411540646078 note
     
    0.27289009446565715 nuforc
     
    0.2714123798568345 pd
     
    0.22600593734740937 orang
     
    0.17383539370291365 star
     
    0.1544049868639051 possibl
     
    0.09934934973598836 like
     
    Topic 6: 
    0.4642862761454686 bright
     
    0.39954375687525034 ufo
     
    0.23988924507500742 shape
     
    0.18503242971605582 sky
     
    0.12707889572744505 craft
     
    0.10855708164395664 triangl
     
    0.06672485824454037 seen
     
    0.06529708501590695 night
     
    0.05630644274962896 cigar
     
    0.0324056520151527 triangular
     
    Topic 7: 
    0.45173969589627094 ufo
     
    0.39678345657207886 red
     
    0.14128190357030534 white
     
    0.13481028604443868 object
     
    0.10238830651130339 flash
     
    0.0940811824313973 seen
     
    0.08807808014284459 firebal
     
    0.07759445016280968 green
     
    0.0652358127940796 fli
     
    0.06325138682753101 like
     
    Topic 8: 
    0.36219687173586856 like
     
    0.3252676019228992 look
     
    0.2943066601109423 craft
     
    0.2886527560141761 star
     
    0.28746447938101066 saw
     
    0.2778931671600617 firebal
     
    0.10074738843324588 north
     
    0.09746901548395433 east
     
    0.09526623081211516 travel
     
    0.09488554237868072 west
     
    Topic 9: 
    0.4744154397483481 red
     
    0.34370831679221986 firebal
     
    0.2159812843755849 shape
     
    0.18404182586423373 bright
     
    0.17195824814305613 white
     
    0.163988035496486 craft
     
    0.15950292250649606 green
     
    0.1549259378306446 hover
     
    0.11619473084812662 orb
     
    0.10987057172824335 glow
     



```python
###Dictionary of Terms and Weights for Topic 0
dicts = {}
for t in range(len(term)):
    dicts[term[t]] = weight[t]
print(dicts)
```

    {'light': 0.5385734656359412, 'sky': 0.18503242971605582, 'bright': 0.18404182586423373, 'object': 0.13481028604443868, 'orang': 0.22600593734740937, 'red': 0.4744154397483481, 'white': 0.17195824814305613, 'shape': 0.2159812843755849, 'flash': 0.10238830651130339, 'fli': 0.0652358127940796, 'ufo': 0.45173969589627094, 'craft': 0.163988035496486, 'like': 0.36219687173586856, 'seen': 0.0940811824313973, 'triangl': 0.10855708164395664, 'triangular': 0.0324056520151527, 'saw': 0.28746447938101066, 'firebal': 0.34370831679221986, 'night': 0.06529708501590695, 'glow': 0.10987057172824335, 'ball': 0.1526174809404168, 'orb': 0.11619473084812662, 'sphere': 0.07456181613951442, 'strang': 0.03075574630383237, 'format': 0.11597383156385913, 'fast': 0.063979813658856, 'disappear': 0.06080654604680967, 'travel': 0.09526623081211516, 'speed': 0.04461121584398084, 'round': 0.03955264519978765, 'sight': 0.27834297124471286, 'note': 0.2758411540646078, 'nuforc': 0.27289009446565715, 'pd': 0.2714123798568345, 'star': 0.2886527560141761, 'possibl': 0.1544049868639051, 'cigar': 0.05630644274962896, 'green': 0.15950292250649606, 'look': 0.3252676019228992, 'north': 0.10074738843324588, 'east': 0.09746901548395433, 'west': 0.09488554237868072, 'hover': 0.1549259378306446}



```python
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
```




    <wordcloud.wordcloud.WordCloud at 0x1a1cfb8c18>




```python
###Term-Concept Matrix
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
concept_term_mat=pd.DataFrame(np.round(svd_model.components_,4),index = [['T'+str(i) for i in range(1,11)]],columns =
terms)
```


```python
###Document-Concept Matrix
doc_concept=pd.DataFrame(np.round(document_topics,3), columns=['T'+str(i) for i in range(1,11)])
doc_concept.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T1</th>
      <th>T2</th>
      <th>T3</th>
      <th>T4</th>
      <th>T5</th>
      <th>T6</th>
      <th>T7</th>
      <th>T8</th>
      <th>T9</th>
      <th>T10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.010</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>-0.006</td>
      <td>-0.002</td>
      <td>0.006</td>
      <td>-0.001</td>
      <td>0.005</td>
      <td>0.012</td>
      <td>0.002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.109</td>
      <td>-0.035</td>
      <td>0.046</td>
      <td>-0.035</td>
      <td>-0.040</td>
      <td>-0.037</td>
      <td>0.010</td>
      <td>-0.005</td>
      <td>0.002</td>
      <td>-0.024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.018</td>
      <td>0.018</td>
      <td>-0.008</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>-0.002</td>
      <td>-0.002</td>
      <td>0.003</td>
      <td>-0.008</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.013</td>
      <td>0.010</td>
      <td>-0.002</td>
      <td>-0.002</td>
      <td>-0.003</td>
      <td>0.005</td>
      <td>-0.001</td>
      <td>0.005</td>
      <td>0.015</td>
      <td>-0.007</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.048</td>
      <td>0.043</td>
      <td>0.014</td>
      <td>0.008</td>
      <td>-0.025</td>
      <td>-0.014</td>
      <td>-0.001</td>
      <td>0.005</td>
      <td>0.026</td>
      <td>-0.073</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Sorting Document-Concept Matrix by highest weighted terms in each topic
doc_concept.sort_values(by='T5',ascending=False).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>T1</th>
      <th>T2</th>
      <th>T3</th>
      <th>T4</th>
      <th>T5</th>
      <th>T6</th>
      <th>T7</th>
      <th>T8</th>
      <th>T9</th>
      <th>T10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79040</th>
      <td>0.403</td>
      <td>0.233</td>
      <td>-0.058</td>
      <td>-0.263</td>
      <td>0.627</td>
      <td>0.217</td>
      <td>0.257</td>
      <td>0.029</td>
      <td>-0.237</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>13065</th>
      <td>0.403</td>
      <td>0.233</td>
      <td>-0.058</td>
      <td>-0.263</td>
      <td>0.627</td>
      <td>0.217</td>
      <td>0.257</td>
      <td>0.029</td>
      <td>-0.237</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>29962</th>
      <td>0.403</td>
      <td>0.233</td>
      <td>-0.058</td>
      <td>-0.263</td>
      <td>0.627</td>
      <td>0.217</td>
      <td>0.257</td>
      <td>0.029</td>
      <td>-0.237</td>
      <td>0.070</td>
    </tr>
    <tr>
      <th>61467</th>
      <td>0.437</td>
      <td>0.126</td>
      <td>0.210</td>
      <td>0.259</td>
      <td>0.572</td>
      <td>0.312</td>
      <td>0.131</td>
      <td>-0.036</td>
      <td>-0.263</td>
      <td>0.064</td>
    </tr>
    <tr>
      <th>33473</th>
      <td>0.437</td>
      <td>0.126</td>
      <td>0.210</td>
      <td>0.259</td>
      <td>0.572</td>
      <td>0.312</td>
      <td>0.131</td>
      <td>-0.036</td>
      <td>-0.263</td>
      <td>0.064</td>
    </tr>
  </tbody>
</table>
</div>




```python
###TOP 5 COMMENTS FOR EACH CONCEPT
columns=list(doc_concept)
x=[doc_concept[column].nlargest(5).index.values for column in doc_concept]
for i in range(0,10):
    for j in range(0,5):
        print('Topic'+ ' ' + str(i+1)+ ' ' +ufo_l[x[i][j]])
###For Each Topic, Printing Comments with Highest Scores
```

    Topic 1 Bright light /Object moving Across sky
    Topic 1 Disk-like&#44 very bright object in the sky with lights
    Topic 1 object in sky with bright light
    Topic 1 Bright light moving in the sky
    Topic 1 bright lights in the sky
    Topic 2 moving light/box shape object
    Topic 2 X shaped object.
    Topic 2 Very LARGE  V shaped objects or objects
    Topic 2 Star/Planet Shaped Flying Object
    Topic 2 us and  the object
    Topic 3 A &quot;break-up&quot; in the sky
    Topic 3 Fire in the sky.
    Topic 3 FIRE IN THE SKY
    Topic 3 Fire in the sky
    Topic 3 fire in the sky
    Topic 4 2 orange fireball&#39s over Bakersfield&#44CA
    Topic 4 Orange lights
    Topic 4 Orange lights
    Topic 4 Orange lights.
    Topic 4 orange lights
    Topic 5 The object  was bright and stauled.
    Topic 5 VERY bright object.
    Topic 5 Bright object.
    Topic 5 Bright Orange Object
    Topic 5 Bright orange object
    Topic 6 Spanaway UFO.  ((NUFORC Note:  Possible sighting of a &quot;twinkling&quot; star?  PD))
    Topic 6 A bright show of A UFO>
    Topic 6 Bright orange lights hovering in the sky.  ((NUFORC Note:  Possible sightings of stars??  PD))
    Topic 6 ufo  or brightest star ever.  ((NUFORC Note:  Possible sighting of stars??  PD))
    Topic 6 Ufo Sighted Very Bright Lights Bristol&#44Connecticut&#44U.S.A.
    Topic 7 A bright show of A UFO>
    Topic 7 Bright light in the sky. UFO or What?
    Topic 7 A Bright light that is a UFO
    Topic 7 a ufo with bright light.
    Topic 7 BRIGHT LIGHTS / UFO
    Topic 8 red ufo
    Topic 8 Nine red - lighted UFO.
    Topic 8 Red and white flashing UFOs.
    Topic 8 Red and white lights on UFO.
    Topic 8 3 Red Glowing UFOs
    Topic 9 I saw what looked like a fireball in the sky.
    Topic 9 It looked like a star at first...
    Topic 9 looks like a star
    Topic 9 Star Like Craft
    Topic 9 I saw what looked like a bright star then it just dimmed out.
    Topic 10 2 red fireballs
    Topic 10 Red fireball..
    Topic 10 Red fireballs
    Topic 10 Five red fireballs.
    Topic 10 A bright red and white glowing fireball in the sky.



```python
###Determining Highest Scored Topic for Each Document
topic_cluster=doc_concept.idxmax(axis=1)
```


```python
###Appending Dominant Topic to Each Document in DataFrame
ufo['topic']=topic_cluster
```


```python
###Viewing Dataset
ufo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>topic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.8830556</td>
      <td>-97.941111</td>
      <td>T9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.38421</td>
      <td>-98.581082</td>
      <td>T1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.2</td>
      <td>-2.916667</td>
      <td>T1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.9783333</td>
      <td>-96.645833</td>
      <td>T9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.4180556</td>
      <td>-157.803611</td>
      <td>T1</td>
    </tr>
  </tbody>
</table>
</div>




```python
###Count of Documents that Belong to Each Topic
ufo['topic'].value_counts()
```


```python
###Reading DataFrame with Dominant Topic Appended to Each Document to CSV
ufo.to_csv(r'/Users/benroberts/Downloads/MSA-Fall1/Python/ufo_topic.csv',index=None)
```
