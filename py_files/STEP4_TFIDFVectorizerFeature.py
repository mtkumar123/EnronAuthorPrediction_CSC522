#!/usr/bin/env python
# coding: utf-8

# In[2]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import numpy as np
import string
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import tree
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif
import matplotlib.pyplot as plt


# In[21]:


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

# In[22]:


df["Folder"].value_counts()


# In[23]:


def uniform_distribution(samples_per_author, df):
    df3 = pd.DataFrame(columns=["Author", "Folder", "File", "Text", "Raw Text"])
    for folder in df["Folder"].value_counts().index:
        df3 = df3.append(df[df["Folder"]==folder].sample(n=samples_per_author), ignore_index=True)
    return df3


# In[24]:


# Change the number of samples per author here
df = uniform_distribution(4000, df)
print(df["Folder"].value_counts())
df


# The function uniform_distribution is used to sample the appropriate number of emails from the number of chosen authors. As you can initially see the number of emails per author is unbalanced with the highest being around 8000 and the lowest being around 4000. To ensure equal distribution we random sampled 4000 emails from each of the 5 authors.

# In[25]:


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


# In[27]:


#Process all the text in each row in the df dataset
proccessed_text = df["Text"].apply(lambda row: text_process(row))
proccessed_text


# In[28]:


#Adding the processed_text as a new column on the df
df["Processed Text"] = proccessed_text


# Here we are doing text preprocessing - remove all non alphanumeric characters, and removing the stop words, and adding that as a column to the dataframe

# In[29]:


#Take the folder and label encode it. That will be our output label (y)
y = df["Folder"]
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)


# Here we are label encoding our target variable Y, which in this case is the folder name or author name of the sent email.

# In[37]:


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
    vectorizer = TfidfVectorizer()
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


# Here we are using the Processed Text and training it with MNB and checking accuracies with 10-Fold Cross Validation. Also in order to select the most relevant features from all the features produced by TFIDF Vectorizer we are using SelectKBest Features. We are searching for the optimal number of K best features by searching from 2000 to 24000 in increments of 2000.
# The average training and testing accuracy list produced contains the accuracy averaged over 10 Folds for 2000 features selected all the way upto 22000 features selected in increments of 2000

# In[9]:


avg_training_accuracy_mnb = [0.8358166666666665, 0.86025, 0.8728444444444443, 0.8792055555555555, 0.8861555555555556, 0.8930944444444444, 0.8954777777777778, 0.8973333333333334, 0.8993999999999998, 0.9013611111111111, 0.903277777777778]
avg_testing_accuracy_mnb = [0.8192999999999999, 0.8375, 0.8446499999999999, 0.8462000000000002, 0.8482, 0.8514999999999999, 0.8530999999999999, 0.8551, 0.8560000000000001, 0.8575999999999999, 0.8593500000000001]

plt.plot(list(range(2000,24000,2000)), avg_training_accuracy_mnb)
plt.plot(list(range(2000,24000,2000)), avg_testing_accuracy_mnb)
plt.xlabel("Number of Features Selected")
plt.ylabel("Accuracy")
plt.title("Select K Best TFIDF MNB")
plt.xticks(range(2000, 24000, 2000))
plt.legend(["Training Accuracy", "Testing Accuracy"])
plt.show()


# Based on this graph we can see the optimal number of Select K best features is K=6000

# In[40]:


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
    vectorizer = TfidfVectorizer()
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


# Here we are using the Processed Text and training it with Random Forests and checking accuracies with 10-Fold Cross Validation. Also in order to select the most relevant features from all the features produced by TFIDF Vectorizer we are using SelectKBest Features. We are searching for the optimal number of K best features by searching from 2000 to 24000 in increments of 2000. The average training and testing accuracy list produced contains the accuracy averaged over 10 Folds for 2000 features selected all the way upto 22000 features selected in increments of 2000

# In[10]:


avg_training_accuracy_rf = [0.9567277777777777, 0.9630777777777777, 0.9656888888888888, 0.9675444444444447, 0.96915, 0.9718944444444446, 0.9729111111111113, 0.973561111111111, 0.9741666666666667, 0.9745944444444443, 0.9751999999999998]
avg_testing_accuracy_rf = [0.8366499999999999, 0.8451000000000001, 0.8485499999999998, 0.8486, 0.8531000000000001, 0.8563000000000001, 0.85595, 0.8543, 0.8574999999999999, 0.8571500000000001, 0.8586]

plt.plot(list(range(2000,24000,2000)), avg_training_accuracy_rf)
plt.plot(list(range(2000,24000,2000)), avg_testing_accuracy_rf)
plt.xlabel("Number of Features Selected")
plt.ylabel("Accuracy")
plt.xticks(range(2000, 24000, 2000))
plt.title("Select K Best TFIDF Random Forests")
plt.legend(["Training Accuracy", "Testing Accuracy"])
plt.show()


# Based on this graph we can see the optimal number of Select K best features is K=4000

# Below we have plotted the graphs for the training and testing accuracies for MNB and Random Forests
# across 5, 10 and 15 authors. RandomForest Classifer gives us the best accuracy

# In[14]:


training_acc =[0.8728444444444443*100,79.56622222222223,73.07953216374269]
testing_acc =[0.8446499999999999*100,75.224,66.8280701754386]


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


# In[17]:


training_acc =[0.9630777777777777*100,95.91111111111112,95.0066276803119]
testing_acc =[0.8451000000000001*100,77.996,70.5719298245614]


labels = [5,10,15]
# ["Random Forests", "MultinomialNB"]
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


# In[42]:


# rf_training_acc =[0.9630777777777777*100,95.91111111111112,95.0066276803119]
rf_testing_acc =[0.8451000000000001*100,77.996,70.5719298245614]
# mnb_training_acc =[0.8728444444444443*100,79.56622222222223,73.07953216374269]
mnb_testing_acc =[0.8446499999999999*100,75.224,66.8280701754386]


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
ax.set_title('TFIDF Accuracies')
ax.set_xticks(x+(2.5*width))
ax.set_xticklabels(labels)
ax.set_xlabel("Number of Authors")

ax.legend()
fig.tight_layout()
plt.rcParams["figure.figsize"] = (6,6)
plt.show()


# In[47]:


# rf_training_acc =[0.9630777777777777*100,95.91111111111112,95.0066276803119]
tf_rf_testing_acc =[0.8451000000000001*100,77.996,70.5719298245614]
# mnb_training_acc =[0.8728444444444443*100,79.56622222222223,73.07953216374269]
tf_mnb_testing_acc =[0.8446499999999999*100,75.224,66.8280701754386]
cv_rf_testing_acc =[0.8423*100,77.36000000000002,69.68070175438597]
cv_mnb_testing_acc =[0.8271*100,73.90800000000002,65.24210526315789]

labels = [5,10,15]
# ["Random Forests", "MultinomialNB"]
x = np.arange(len(labels))
length = np.arange(len(labels))
width = 0.2
fig, ax = plt.subplots()
rects1 = ax.bar(x + width, tf_rf_testing_acc, width, label='TFIDF Random Forest')
rects2 = ax.bar(x + 2*width, cv_rf_testing_acc, width, label='CV Random Forest')
rects3 = ax.bar(x + 3*width, tf_mnb_testing_acc, width, label='TFIDF MNB')
rects4 = ax.bar(x + 4*width, cv_mnb_testing_acc, width, label='CV MNB')
ax.set_ylabel('Accuracies')
ax.set_title('CountVec vs TFIDF Accuracies')
ax.set_xticks(x+(2.5*width))
ax.set_xticklabels(labels)
ax.set_xlabel("Number of Authors")

ax.legend()
fig.tight_layout()
plt.rcParams["figure.figsize"] = (8,6)
plt.show()


# In[ ]:
