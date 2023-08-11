#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[3]:


for dirname, _, filenames in os.walk('../'):
    print(filenames)
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


df = pd.read_csv("../input/enron29features/Enron_29_Features.csv")


# This notebook was actually run on kaggle, as K-means clustering was not working on our local machines. Here we are using the Enron 29 Feature Dataset that we created extracting all the stylometric features from all the author emails found in the initial dataset Enron.csv

# In[26]:


import re
import string
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter 
from sklearn.linear_model import LogisticRegression
from gensim.models import Word2Vec

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
nltk.download("stopwords")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn import preprocessing
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf
import string
import re
from collections import Counter


# In[27]:


from nltk.corpus import stopwords
stop_words = stopwords.words("english")


# In[28]:


df["Folder"].value_counts()


# Here we are testing with 5 authors, 4000 emails per author

# In[29]:


def text_process(text):
    #Remove Punctuation Marks
    text = text.lower()
    nopunct = ""
    clean_final = []
    for char in text:
        if re.match(r"\w", char) or re.match(r" ", char):
            nopunct += char
        elif re.match(r" ", char):
            nopunct += char
        else:
            nopunct += " "
    for word in nopunct.split():
        if not word in stop_words:
            clean_final.append(word)
    nopunct=" ".join(clean_final)
    return nopunct


# In[30]:


df = df[df["Email Length"].notna()]
df = df[df["Author"].notna()]
clean_text = df["Text"].apply(lambda row: text_process(row))
df["Processed Text"] = clean_text


# Here we performed text preprocessing, removing all the stop words, and non alphanumeric characters

# 

# In[47]:


df["Processed Text"] = proccessed_text


# In[48]:


tokenized_text = df["Processed Text"].apply(lambda row: word_tokenize(row))
df["Tokens"] = tokenized_text


# In[33]:


def vectorize(list_of_docs, model):

    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


# This function takes input of a list of tokens, a trained word2vec model. The function then goes through all the tokens and finds the appropriate word2vec vector and appends the found vector to a list called vectors. To represent the all the word vectors present in the list vectors, we take the average of the vectors and return that to represent the list of tokens passed into the function. If none of the tokens passed in were found in the glove word dictionary a zero vector is returned.

# 

# In[34]:


X = df["Tokens"].to_numpy()
y = df["Folder"]
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# Here we are label encoding our target variable Y, which in this case is the folder name or author name of the sent email.

# 

# In[35]:


X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33)

tokenized_docs = X_train["Tokens"]
model = Word2Vec(sentences=tokenized_docs, vector_size=100, workers=1, seed=42)
vectorized_docs = vectorize(tokenized_docs, model=model)

test_tokenized_docs = X_test["Tokens"]
test_vectorized_docs = vectorize(test_tokenized_docs, model=model)


# In[36]:


print(X_train.shape)
vect_np = np.asarray(vectorized_docs)
print(vect_np.shape)
print(X_test.shape)
vect_test_np = np.asarray(test_vectorized_docs)
print(vect_test_np.shape)


# In[37]:


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            ed = euclidean_distances(points[i].reshape(1, -1), curr_center.reshape(1, -1))
            curr_sse += ed
        sse.append(curr_sse)
    return sse


# In[38]:


wss = calculate_WSS(vect_np, 30)


# Here we are using WSS, to try to find the optimal number of K

# In[53]:


print(wss)
y_axis = []
for _ in wss:
    y_axis.append(_[0][0])
x_axis = list(range(1,31,1))
plt.plot(x_axis, y_axis)
plt.xlabel("Number of Clusters")
plt.show()


# However, looking at our graph there is no clear elbow point to choose the right number of clustes

# In[40]:


def best_silhouette_score(points, kmax):
    sil = []
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters = k).fit(points)
        labels = kmeans.labels_
        score = silhouette_score(points, labels, metric = 'euclidean')
        print(score)
        sil.append(score)
    return sil


# In[41]:


sil_scores = best_silhouette_score(vect_np, 30)


# Let's try to use the Silhoutte Method to find the right number of clusters

# In[54]:


print(sil_scores)
x_axis = list(range(2,31,1))
plt.plot(x_axis, sil_scores)
plt.xlabel("Number of Clusters")
plt.show()


# But here also there is no clear peak with which we can choose the Number of Clusters. Let's explore why clustering is not a good option here, with k=30.

# In[43]:


kmeans = KMeans(n_clusters = 30).fit(vect_np)
training_labels = kmeans.labels_
testing_labels = kmeans.predict(vect_test_np)


# In[44]:


for i in range(0,30,1):
    cluster_df = df.iloc[np.where(training_labels==i)]
    print(cluster_df.shape)
    print("Round {}".format(i))
cluster_df[["Tokens", "Processed Text", "Folder"]]


# It looks like the data is getting pretty nicely distributed amongst the 30 clusters. Here clustering is based on the vectors of the emails - which is the average of all the word vectors contained in each email. Let's look at the emails found within the last cluster. As you can see from the Processed Text none of the emails really have anything in common, they show no proper relationship or meaningful insight. 
# We believe that by taking the average of all the word vectors and using that to represent the overall email lost a lot of information, and thus clustering using the resulting vector did not result in meaningful clusters.

# In[ ]:





# In[ ]:




