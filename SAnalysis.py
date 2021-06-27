#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras


# In[2]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


# In[4]:


movie_reviews = pd.read_csv("C:/Desktop/Sentiment_analysis/IMDB Dataset.csv")

movie_reviews.isnull().values.any()

movie_reviews.shape


# In[5]:


movie_reviews.head()


# In[9]:


movie_reviews["review"][1]


# In[10]:


import seaborn as sns

sns.countplot(x='sentiment', data=movie_reviews)


# In[11]:


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


# In[12]:


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


# In[13]:


X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))


# In[14]:


X[0]


# In[15]:


y = movie_reviews['sentiment']

y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


# In[17]:


y_train[0]


# In[18]:


X_train[0]


# In[19]:


X_train[1]


# In[20]:


y_train[1]


# In[49]:


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train + X_test)


# In[50]:


tokenizer.word_index


# In[51]:


len(tokenizer.word_index)


# In[52]:


X_train_t = tokenizer.texts_to_sequences(X_train)
X_test_t = tokenizer.texts_to_sequences(X_test)


# In[53]:


np.array(X_train_t[1])


# In[31]:


len(X_train_t[40000])


# In[54]:


len(X_train_t[39999])


# In[59]:


len(X_train_t[1])


# In[56]:


print(X_train[0])


# In[57]:


check = X_train[1].split()


# In[58]:


print(len(check))


# In[60]:


num_tokens = [len(tokens) for tokens in X_train_t + X_test_t]
num_tokens = np.array(num_tokens)


# In[62]:


np.mean(num_tokens)


# In[63]:


np.max(num_tokens)


# In[64]:


np.min(num_tokens)


# In[65]:


vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train_pad = pad_sequences(X_train_t, padding='post', maxlen=maxlen)
X_test_pad = pad_sequences(X_test_t, padding='post', maxlen=maxlen)


# In[66]:


np.array(X_train_pad[1])


# In[67]:


len(np.array(X_train_pad[1]))


# In[72]:


len(X_train_t[900])


# In[73]:


np.array(X_train_pad[900])


# In[74]:


len(np.array(X_train_pad[1]))


# In[76]:


top_words = 5000


# In[79]:


from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# In[80]:


model = Sequential()
model.add(Embedding(top_words, 32, input_length=maxlen))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[126]:


model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=6, batch_size=128, verbose=1,validation_split=0.2)
# Final evaluation of the model
scores = model.evaluate(X_test_pad, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[133]:


instance = "not a good movie"


# In[134]:


instance = tokenizer.texts_to_sequences(instance)


# In[135]:


flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)


# In[136]:


flat_list = [flat_list]


# In[137]:


instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)


# In[138]:


model.predict(instance)


# In[140]:


model = Sequential()


model.add(Embedding(top_words, 32, input_length=maxlen))

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


# In[142]:


history = model.fit(X_train_pad, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

score = model.evaluate(X_test_pad, y_test, verbose=1)


# In[143]:


print(X[57])


# In[149]:


instance = ' good'
instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

model.predict(instance)


# In[ ]:




