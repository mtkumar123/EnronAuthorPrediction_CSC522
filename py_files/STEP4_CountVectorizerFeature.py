#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import string
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


# In[2]:


#Put in the file path to the dataset created from extractingauthors.ipynb
df = pd.read_csv("./enron.csv")
df = df.drop(["Email Folder"], axis=1)
#We need only the top 20 authors ordered by number of emails found in either the
#sent folder or _sent_mail folder

#Add top_authors = df.value_counts(["Folder"])[:X] for the number of authors required
# Change X to 5,10,15 to test with 5, 10, 15 authors
top_authors = df.value_counts(["Folder"])[:5]
df = df.loc[df["Folder"].isin(list(top_authors.index.get_level_values(0)))].drop(["Unnamed: 0"], axis=1).reset_index(drop=True)
df = df[df["Text"]!=" "]
df = df[df["Text"]!="\n"]
df = df.dropna()


# Here we are using the enron.csv file created from the noteboook "STEP2_ExtractingAuthorEmails.ipynb". This csv contains all the sent emails from 20 authors, and the extracted body of text from each email. You can change the X value too 5,10,15 to test with the corresponding number of authors. Here we are testing with 5 authors.

# In[3]:


df["Folder"].value_counts()


# In[4]:


def uniform_distribution(samples_per_author, df):
    df3 = pd.DataFrame(columns=["Author", "Folder", "File", "Text", "Raw Text"])
    for folder in df["Folder"].value_counts().index:
        df3 = df3.append(df[df["Folder"]==folder].sample(n=samples_per_author), ignore_index=True)
    return df3


# In[5]:


# Change the number of samples per author here
df = uniform_distribution(4000, df)
print(df["Folder"].value_counts())
df


# The function uniform_distribution is used to sample the appropriate number of emails from the number of chosen authors. As you can initially see the number of emails per author is unbalanced with the highest being around 8000 and the lowest being around 4000. To ensure equal distribution we random sampled 4000 emails from each of the 5 authors.

# In[6]:


#Here I am creating a basic text processing function, to do text cleaning
#lemmatization, stop word removal.

def text_process(text):
    lemmatiser = WordNetLemmatizer()
    #Remove Punctuation Marks
    nopunct=[char for char in text if char not in string.punctuation]
    nopunct=''.join(nopunct)
    #Lemmatisation
    a=''
    i=0
    for i in range(len(nopunct.split())):
        b=lemmatiser.lemmatize(nopunct.split()[i], pos="v")
        a=a+b+' '
    #Removal of Stopwords
    words = ""
    for word in a.split():
        if word.lower():
            word = word.lower()
            if word not in stopwords.words("english"):
                words = words + word + " "

    return words


# In[7]:


#Process all the text in each row in the df dataset
proccessed_text = df["Text"].apply(lambda row: text_process(row))
proccessed_text


# In[10]:


#Adding the processed_text as a new column on the df
df["Processed Text"] = proccessed_text


# Here we are doing text preprocessing - remove all non alphanumeric characters, and removing the stop words, and adding that as a column to the dataframe

# In[11]:


#Take the folder and label encode it. That will be our output label (y)
y = df["Folder"]
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# Here we are label encoding our target variable Y, which in this case is the folder name or author name of the sent email.

# In[12]:


#X input variable will the processed text. Convert to numpy.

#Testing with Naive Bayes first
X = df["Processed Text"].to_numpy()
kf = KFold(n_splits=10, random_state=1, shuffle=True)
model = MultinomialNB()
training_accuracies = {k: [] for k in range(2000,24000,2000)}
testing_accuracies = {k: [] for k in range(2000,24000,2000)}
number_features = range(2000, 24000, 2000)
avg_training_accuracy = []
avg_testing_accuracy = []

i = 1
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    vectorizer = CountVectorizer()
    transformer = vectorizer.fit(X_train)

    X_train_transformed = transformer.transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    for k in range(2000,24000,2000):
        print("Round {} of CV Testing".format(i))
        print("Number of Features for SelectKBest = {}".format(k))
        fs = SelectKBest(k=k)
        X_train_transformed_new = fs.fit_transform(X_train_transformed, y_train)
        X_test_transformed_new = fs.transform(X_test_transformed)

        model.fit(X_train_transformed_new, y_train)
        training_accuracy = model.score(X_train_transformed_new, y_train)
        testing_accuracy = model.score(X_test_transformed_new, y_test)
        print(training_accuracy)
        print(testing_accuracy)
        training_accuracies[k].append(training_accuracy)
        testing_accuracies[k].append(testing_accuracy)
    i = i + 1
#     training_accuracies.append(training_accuracy)
#     testing_accuracies.append(testing_accuracy)
for k in training_accuracies.keys():
    avg_training_accuracy.append(sum(training_accuracies[k])/len(training_accuracies[k]))
for k in testing_accuracies.keys():
    avg_testing_accuracy.append(sum(testing_accuracies[k])/len(testing_accuracies[k]))

#Printing out average training accuracy for each number of features selected 2000,4000,etc..
print("Average Training Accuracy")
print(avg_training_accuracy)
print("Average Testing Accuracy")
print(avg_testing_accuracy)


# Here we are using the Processed Text and training it with MNB and checking accuracies with 10-Fold Cross Validation. Also in order to select the most relevant features from all the features produced by Count Vectorizer we are using SelectKBest Features. We are searching for the optimal number of K best features by searching from 2000 to 24000 in increments of 2000. The average training and testing accuracy list produced contains the accuracy averaged over 10 Folds for 2000 features selected all the way upto 22000 features selected in increments of 2000

# In[17]:


avg_training_accuracy_mnb = [0.8139444444444444, 0.8356722222222223, 0.8460722222222221, 0.8547944444444445, 0.8604277777777778, 0.8662222222222222, 0.8709555555555555, 0.8722277777777776, 0.8736222222222223, 0.8753833333333333, 0.8776833333333334]
avg_testing_accuracy_mnb = [0.8032499999999999, 0.81835, 0.8271, 0.8308500000000001, 0.8335999999999999, 0.8368, 0.8408999999999999, 0.84215, 0.84275, 0.8442500000000001, 0.8469499999999999]

plt.plot(list(range(2000,24000,2000)), avg_training_accuracy_mnb)
plt.plot(list(range(2000,24000,2000)), avg_testing_accuracy_mnb)
plt.xlabel("Number of Features Selected")
plt.ylabel("Accuracy")
plt.title("Select K Best Count Vectorizer MNB")
plt.xticks(range(2000, 24000, 2000))
plt.legend(["Training Accuracy", "Testing Accuracy"])
plt.show()


# Based on this graph we can see the optimal number of Select K best features is K=6000

# In[15]:


#X input variable will the processed text. Convert to numpy.

#Testing with Random Forests
X = df["Processed Text"].to_numpy()
kf = KFold(n_splits=10, random_state=1, shuffle=True)
model = RandomForestClassifier()
training_accuracies = {k: [] for k in range(2000,24000,2000)}
testing_accuracies = {k: [] for k in range(2000,24000,2000)}
number_features = range(2000, 24000, 2000)
avg_training_accuracy = []
avg_testing_accuracy = []

i = 1
for train_index, test_index in kf.split(X):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    vectorizer = CountVectorizer()
    transformer = vectorizer.fit(X_train)

    X_train_transformed = transformer.transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    for k in range(2000,24000,2000):
        print("Round {} of CV Testing".format(i))
        print("Number of Features for SelectKBest = {}".format(k))
        fs = SelectKBest(k=k)
        X_train_transformed_new = fs.fit_transform(X_train_transformed, y_train)
        X_test_transformed_new = fs.transform(X_test_transformed)

        model.fit(X_train_transformed_new, y_train)
        training_accuracy = model.score(X_train_transformed_new, y_train)
        testing_accuracy = model.score(X_test_transformed_new, y_test)
        print(training_accuracy)
        print(testing_accuracy)
        training_accuracies[k].append(training_accuracy)
        testing_accuracies[k].append(testing_accuracy)
    i = i + 1
#     training_accuracies.append(training_accuracy)
#     testing_accuracies.append(testing_accuracy)
for k in training_accuracies.keys():
    avg_training_accuracy.append(sum(training_accuracies[k])/len(training_accuracies[k]))
for k in testing_accuracies.keys():
    avg_testing_accuracy.append(sum(testing_accuracies[k])/len(testing_accuracies[k]))

#Printing out average training accuracy for each number of features selected 2000,4000,etc..
print("Average Training Accuracy")
print(avg_training_accuracy)
print("Average Testing Accuracy")
print(avg_testing_accuracy)


# Here we are using the Processed Text and training it with Random Forests and checking accuracies with 10-Fold Cross Validation. Also in order to select the most relevant features from all the features produced by Count Vectorizer we are using SelectKBest Features. We are searching for the optimal number of K best features by searching from 2000 to 24000 in increments of 2000. The average training and testing accuracy list produced contains the accuracy averaged over 10 Folds for 2000 features selected all the way upto 22000 features selected in increments of 2000

# In[18]:


avg_training_accuracy_rf = [0.9481777777777778, 0.9599611111111113, 0.9638277777777778, 0.9667555555555556, 0.9686944444444444, 0.97055, 0.9722444444444445, 0.9727499999999999, 0.9733444444444445, 0.9739722222222221, 0.9746999999999998]
avg_testing_accuracy_rf = [0.8322499999999999, 0.8423, 0.8442999999999999, 0.8462499999999998, 0.84765, 0.84885, 0.8511, 0.8516, 0.8507, 0.8516, 0.85225]

plt.plot(list(range(2000,24000,2000)), avg_training_accuracy_rf)
plt.plot(list(range(2000,24000,2000)), avg_testing_accuracy_rf)
plt.xlabel("Number of Features Selected")
plt.ylabel("Accuracy")
plt.title("Select K Best Count Vectorizer Random Forests")
plt.xticks(range(2000, 24000, 2000))
plt.legend(["Training Accuracy", "Testing Accuracy"])
plt.show()


# Based on this graph we can see the optimal number of Select K best features is K=4000

# Here we have plotted the graphs for the training and testing accuracies for MNB and Random Forests
# across 5, 10 and 15 authors. RandomForest Classifer gives us the best accuracy

# In[31]:


training_acc =[0.8460722222222221*100,77.38977777777778,69.88187134502925]
testing_acc =[0.8271*100,73.90800000000002,65.24210526315789]



labels = [5,10,15]
x = np.arange(len(labels))
length = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_acc, width, label='Training Accuracies')
rects2 = ax.bar(x + width/2, testing_acc, width, label='Testing Accuracies')
ax.set_ylabel('Accuracies')
ax.set_title('MultinomialNB Accuracies')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Number of Authors")

ax.legend()
fig.tight_layout()
plt.rcParams["figure.figsize"] = (8,8)

plt.show()


# In[34]:


training_acc =[0.9599611111111113*100,95.39555555555556,94.18635477582846]
testing_acc =[0.8423*100,77.36000000000002,69.68070175438597]



labels = [5,10,15]
x = np.arange(len(labels))
length = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, training_acc, width, label='Training Accuracies')
rects2 = ax.bar(x + width/2, testing_acc, width, label='Testing Accuracies')
ax.set_ylabel('Accuracies')
ax.set_title('Random Forests Accuracies')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel("Number of Authors")

ax.legend()
fig.tight_layout()
plt.rcParams["figure.figsize"] = (12,12)
plt.show()


# In[40]:


rf_testing_acc =[0.8423*100,77.36000000000002,69.68070175438597]
mnb_testing_acc =[0.8271*100,73.90800000000002,65.24210526315789]


labels = [5,10,15]
# ["Random Forests", "MultinomialNB"]
x = np.arange(len(labels))
length = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots()
# rects1 = ax.bar(x + width, mnb_training_acc, width, label='MNB Training Accuracies')
rects2 = ax.bar(x + 2*width, mnb_testing_acc, width, label='MNB Testing Accuracies')
# rects3 = ax.bar(x + 3*width, rf_training_acc, width, label='RF Training Accuracies')
rects4 = ax.bar(x + 3*width, rf_testing_acc, width, label='RF Testing Accuracies')
ax.set_ylabel('Accuracies')
ax.set_title('CountVec Accuracies')
ax.set_xticks(x+(2.5*width))
ax.set_xticklabels(labels)
ax.set_xlabel("Number of Authors")

ax.legend()
fig.tight_layout()
plt.rcParams["figure.figsize"] = (7,7)
plt.show()


# In[ ]:
