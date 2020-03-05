
# Machine Learning Translation: English to French Using RNNs


## Data Preprocessing


```python
# Importing NLTK and Libraries
import nltk
import string
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.models import Sequential
```


```python
# Importing Corpus 
from nltk.corpus import PlaintextCorpusReader
from google.colab import drive
drive.mount('/content/drive')
corpus_root = '/content/drive/My Drive/en-fr'
wordlists = PlaintextCorpusReader(corpus_root,'.*')
wordlists.fileids()

```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).





    ['small_vocab_en.txt', 'small_vocab_fr.txt']




```python
# Assigning English and French Text
en_text=wordlists.raw('small_vocab_en.txt')
fr_text=wordlists.raw('small_vocab_fr.txt')
```


```python
# Splitting Texts into Individual Sentences
en_text = en_text.split('\n')
fr_text = fr_text.split('\n')
#print(en_text[0:3])
#print(sp_text[0:3])
```


```python
# Splitting into list of lists of words
en_text = [sent.split(' ') for sent in en_text]
fr_text = [sent.split(' ') for sent in fr_text]

# Removing Punctuation
table = str.maketrans('','', string.punctuation)
en_text = [[word.translate(table) for word in sent] for sent in en_text]
fr_text = [[word.translate(table) for word in sent] for sent in fr_text]

#print(en_text[0:3])
#print('\n')
#print(sp_text[0:3])
```


```python
# Compute length of English sentences
en_sent_lengths = [len(en_sent) for en_sent in en_text]
# Compute the max English sentence length
en_len = int(round(np.max(en_sent_lengths)))
print('(English) Mean sentence length: ', en_len)

# Compute length of French sentences
fr_sent_lengths = [len(fr_sent) for fr_sent in fr_text]
# Compute the max French sentence length
fr_len = int(round(np.max(fr_sent_lengths)))
print('(Spanish) Mean sentence length: ', fr_len)

# Compute length of English vocabulary
en_words = []
for sent in en_text:
  # Populate all_words with all the words in sentences
  en_words.extend(sent)
# Compute the length of the set containing all_words
en_vocab = len(set(en_words))
print("(English) Vocabulary size: ", en_vocab)

# Compute length of French vocabulary
fr_words = []
for sent in fr_text:
  # Populate all_words with all the words in sentences
  fr_words.extend(sent)
# Compute the length of the set containing all_words
fr_vocab = len(set(fr_words))
print("(Spanish) Vocabulary size: ", fr_vocab)

```

    (English) Mean sentence length:  17
    (Spanish) Mean sentence length:  23
    (English) Vocabulary size:  200
    (Spanish) Vocabulary size:  346



```python
#Tokenizing Texts
en_tok = Tokenizer()
en_tok.fit_on_texts(en_text)

fr_tok = Tokenizer()
fr_tok.fit_on_texts(fr_text)

# Combine text in each list into a full sentences
en_sentences = [' '.join(sent) for sent in en_text]
fr_sentences = [' '.join(sent) for sent in fr_text]

```


```python
# Convert Each English Sentence into a Sequence Vector the Length of the Longest French Sentence
def sents2seqs(sentences, pad_type='post'):
    # Each Word in the Sentence is Converted to a Number Representing its Rank in the Tokenized Vocabulary     
    encoded_text = en_tok.texts_to_sequences(sentences)
    # Pad Each Sentence with 0s so that Each Vector is the Same Length
    preproc_text = pad_sequences(encoded_text, padding=pad_type, truncating='post', maxlen=fr_len)
    return preproc_text
    
en_x = sents2seqs(en_sentences)
#en_x=en_x.reshape(*en_x.shape, 1)
#print(en_x.shape)
```


```python
# Convert Each French Sentence into a Sequence Vector the Length of the Longest French Sentence
def sents2seqs_fr(sentences, pad_type='post'):     
    encoded_text = fr_tok.texts_to_sequences(sentences)
    preproc_text = pad_sequences(encoded_text, padding=pad_type, truncating='post', maxlen=fr_len)
    return preproc_text

fr_y = sents2seqs_fr(fr_sentences)

# Reshape French and English Vectors to Input into the Model
fr_y=fr_y.reshape(*fr_y.shape, 1)
en_x=en_x.reshape((-1, fr_y.shape[-2]))
```

## Defining and Training the Embedded RNN Model




```python
# Model Function
def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3

    # Setting Up Layers
    rnn = GRU(64, return_sequences=True, activation="tanh")
    embedding = Embedding(french_vocab_size, 64, input_length=input_shape[1]) 
    logits = TimeDistributed(Dense(french_vocab_size+1, activation="softmax"))
    
    model = Sequential()
    model.add(embedding)
    model.add(rnn)
    model.add(logits)
    
    # Loss Function
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model


# Training and Evaluating Model
embedded_model = embed_model(
    en_x.shape,
    fr_len,
    en_vocab,
    fr_vocab)
embedded_model.fit(en_x, fr_y, batch_size=1024, epochs=10, validation_split=0.2)

```

    Train on 110288 samples, validate on 27573 samples
    Epoch 1/10
    110288/110288 [==============================] - 58s 528us/step - loss: 3.5736 - acc: 0.4511 - val_loss: 2.7546 - val_acc: 0.4641
    Epoch 2/10
    110288/110288 [==============================] - 55s 497us/step - loss: 2.5123 - acc: 0.4808 - val_loss: 2.2108 - val_acc: 0.5468
    Epoch 3/10
    110288/110288 [==============================] - 55s 496us/step - loss: 1.8630 - acc: 0.5940 - val_loss: 1.5829 - val_acc: 0.6305
    Epoch 4/10
    110288/110288 [==============================] - 55s 497us/step - loss: 1.4191 - acc: 0.6566 - val_loss: 1.2644 - val_acc: 0.6917
    Epoch 5/10
    110288/110288 [==============================] - 55s 496us/step - loss: 1.1611 - acc: 0.7186 - val_loss: 1.0666 - val_acc: 0.7408
    Epoch 6/10
    110288/110288 [==============================] - 55s 502us/step - loss: 1.0012 - acc: 0.7526 - val_loss: 0.9357 - val_acc: 0.7662
    Epoch 7/10
    110288/110288 [==============================] - 55s 500us/step - loss: 0.8844 - acc: 0.7767 - val_loss: 0.8309 - val_acc: 0.7886
    Epoch 8/10
    110288/110288 [==============================] - 55s 497us/step - loss: 0.7891 - acc: 0.7959 - val_loss: 0.7454 - val_acc: 0.8042
    Epoch 9/10
    110288/110288 [==============================] - 55s 497us/step - loss: 0.7119 - acc: 0.8109 - val_loss: 0.6786 - val_acc: 0.8184
    Epoch 10/10
    110288/110288 [==============================] - 55s 496us/step - loss: 0.6537 - acc: 0.8225 - val_loss: 0.6325 - val_acc: 0.8254





    <keras.callbacks.History at 0x7f4673ce7160>



## Translating an English Sentence to French


```python
def translate(sent):
    #English Sentence
    sentence = sent
    #Converting to Sequence Vector
    en_seq = sents2seqs(sentence)[0]
    en_seq=en_seq.reshape((-1, fr_y.shape[-2]))
    #Predicting French Sequence Vector
    logits = embedded_model.predict(en_seq)[0]

    # Creating Index of Words in Vocabulary to Convert Predicted Sequences to French Words
    index_to_words = {id: word for word, id in fr_tok.word_index.items()}
    index_to_words[0] = '<PAD>'
    #Return French Translation
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print(translate(['it is hot today']))
```

    il est est chaud en <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD> <PAD>

