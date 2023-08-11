#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import spacy
from collections import Counter
import copy
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_lg")


# Using the enron.csv file created in the previous notebook ,
# we are extracting the emails of only top 5,10,15 authors(sorted according to no of emails per author) for our analysis 
# and dropping any mails that have no text.

# In[55]:


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


# In[56]:


df["Folder"].value_counts()


# Random Sampling equal number of emails from each author 

# In[19]:


def uniform_distribution(samples_per_author, df):
    df3 = pd.DataFrame(columns=["Author", "Folder", "File", "Text", "Raw Text"]) 
    for folder in df["Folder"].value_counts().index:
        df3 = df3.append(df[df["Folder"]==folder].sample(n=samples_per_author), ignore_index=True)
    return df3


# In[58]:


# Change the number of samples per author here
df = uniform_distribution(4000, df)
print(df["Folder"].value_counts())
df


# In the next three cells , we have written function to extract the stylometric features(a combination of lexical
# , structural and syntatic features)of a particular email using regex matching.

# In[27]:


#Feature Extraction - Manoj
#extract feature - email length in characters. Exclude all whitespace. 
def feature_email_length_characters(text):
    if type(text) == str:
        text = text.strip()
        text = re.sub(r"\W", "", text)
        length = len(text)
        return length
    else:
        return np.NaN

#extract digit density - ratio of number of digits to number of characters
def feature_digit_density(text):
    if type(text) == str:
        text = text.strip()
        text = re.sub(r"\W", "", text)
        total_length = len(text)
        number_digits = len(re.findall(r"\d", text))
        try:
            return (number_digits/total_length)
        except ZeroDivisionError as e:
            return 0
    if type(text) == float:
        return np.NaN

#extract space density - ratio of space to number of characters
def feature_space_density(text):
    if type(text) == str:
        number_space = len(re.findall(r"[\s\n\t]", text))
        text = text.strip()
        text = re.sub(r"\W", "", text)
        total_length = len(text)
        try:
            return (number_space/total_length)
        except ZeroDivisionError as e:
            return 0
    if type(text) == float:
        return np.NaN

#extract number of paragraphs
def feature_paragraph(text):
    if type(text) == str:
        number_paragraphs = len(re.findall(r"\n\n", text))
        return max(1.0, number_paragraphs)
    if type(text) == float:
        return np.NaN
    
# extract number of sentences in paragraphs
def feature_average_characters_paragraph(text):
    if type(text) == str:
        number_paragraphs = len(re.findall(r"\n\n", text))
        if not number_paragraphs:
            return max(1, len(re.findall(r"[.?!]\W", text)))
        else:
            paragraphs = re.findall(r"(?:.+\n)+\n", text)
            length = 0
            for paragraph in paragraphs:
                length += max(1, len(re.findall(r"[.?!]\W", paragraph)))
            return length/number_paragraphs
    if type(text) == float:
        return np.NaN
    

#extract farewell words
def feature_farewell_words(text):
    if type(text) == str:
        try:
            words = text.split()
            for word in reversed(words):
                if re.search(r"\w+", word):
                    last_word = re.search(r"\w+", word).group().lower()
                    return last_word
                else:
                    continue
            return "\n"
        except IndexError as e:
            return np.NaN
    if type(text) == float:
        return np.NaN

def feature_freq_farewell_words(farewell_words, text):
    if type(text) == str:
        if text in farewell_words:
            return text
        else:
            return "Other"
    if type(text) == float:
        return np.NaN
        

#extract last punctuation
def feature_ending_punctuation(text):
    if type(text) == str:
        if re.search(r"[\!\,\.\?\:\'\"]", text):
            try:
                last_punc = re.findall(r"[\!\,\.\?\:\'\"]", text)[-1]
                return last_punc
            except IndexError as e:
                last_punc = len(re.findall(r"[\!\,\.\?\:\'\"]", text))
                if len == 1:
                    return last_punc[0]
        else:
            return "None"
    if type(text) == float:
        return np.NaN

#extract most common used punctuation in the email
def feature_most_used_punctuation(text):
    if type(text) == str:
        if re.search(r"[\!\,\.\?\:\'\"]", text):
            punc = re.findall(r"[\!\,\.\?\:\'\"]", text)
            most_used_punc = Counter(punc).most_common(1)[0][0]
            return most_used_punc
        else:
            return "None"
    if type(text) == float:
        return np.NaN
    
#extract subjectivity and polarity
def feature_subjectivity(text):
    if type(text) == str:
        blob = TextBlob(text)
        return blob.sentiment.subjectivity
    if type(text) == float:
        return np.NaN

#extract polarity
def feature_polarity(text):
    if type(text) == str:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    if type(text) == float:
        return np.NaN

def feature_most_pos(text):
    if type(text) == str:
        blob = TextBlob(text)
        final_pos = []
        for word, pos in blob.tags:
            if word not in stopwords.words("english"):
                final_pos.append(pos)
        count_pos = Counter(final_pos)
        if count_pos.most_common():
            return count_pos.most_common()[0][0]
        else:
            return "Other"
    if type(text) == float:
        return np.NaN

def feature_get_greeting(text):
    if type(text) == str:
        if re.match(r"^\w+", text):
            greeting_word = re.match(r"^\w+", text).group()
            return greeting_word
        else:
            return "None"
    if type(text) == float:
        return np.NaN

def feature_most_common_word(text):
    if type(text) == str:
        blob = TextBlob(text)
        words = []
        for word, pos in blob.tags:
            words.append(word)
        count_word = Counter(words)
        if count_word.most_common():
            return count_word.most_common()[0][0]
        else:
            return np.NaN
    if type(text) == float:
        return np.NaN

def feature_freq_most_common_word(text):
    if type(text) == str:
        blob = TextBlob(text)
        words = []
        for word, pos in blob.tags:
            words.append(word)
        count_word = Counter(words)
        if count_word.most_common():
            return count_word.most_common()[0][1]
        else:
            return 0
    if type(text) == float:
        return np.NaN

def feature_number_words(text):
    if type(text) == str:
        blob = TextBlob(text)
        return len(blob.words)
    if type(text) == float:
        return np.NaN


# In[28]:


#Feature Extraction - Sairaj
#Average length of words
def avg_length(text):
    if type(text) == str:
        list1 = text.split()
        word_len = 0
        for text in list1:
            strip_text=text.strip()
            if re.search(r"\w+",strip_text):
                strip_text = re.search(r"\w+",strip_text).group()
                word_len += len(strip_text)
        try:
            avg_word_len = word_len/len(list1)
        except ZeroDivisionError as e:
            return np.NaN
        return avg_word_len
    else:
        return np.NaN

#Average Sentence Length
def avg_sentence_length(text):
    if type(text) == str:
        list1 = re.findall(r"[^\.\?\!]+",text)
        sent_len = 0 
        for le in list1:
            sent_len += len(le)
        try:
            return sent_len/len(list1) 
        except:
            return np.NaN      
    else:
        return np.NaN

#Number of short words to overall number of words
def feature_short_word_ratio(text):
    if type(text) == str:
        list1 = text.split()
        short_word = 0
        if len(list1) >= 1:
            for word in list1:
                strip_text = word.strip()
                if re.search(r"\w+",strip_text):
                    strip_text = re.search(r"\w+",strip_text).group()
                    word_len = len(strip_text)
                    if word_len < 4:
                        short_word +=1
            try:
                short_word_rat = short_word/len(list1)
                return short_word_rat
            except:
                return 0
        else:
            return 0  
    else:
        return np.NaN

#Frequency of punctuation
def punctuation_frequency(text):
    if type(text) == str:
        if re.search(r"[\!\,\.\?\:\'\"]", text):
            punc = re.findall(r"[\!\,\.\?\:\'\"]", text)
            freq_punc = len(punc)
            return freq_punc
        else:
            return 0
    if type(text) == float:
        return np.NaN

#Punctuation after greeting
def punctuation_greeting(text):
    if type(text) == str:
        if re.search(r"^\w+([\,\:\?\!\-])\n", text):
            punc = re.search(r"^\w+([\,\:\?\!\-])\n", text).group(1)
            return punc
        else:
            return "None"
    if type(text) == float:
        return np.NaN

def feature_number_special_characters(text):
    if type(text) == str:
        special_characters = re.findall(r"[\@\#\$\%\^\&\~\`\*\(\)\<\>\\\[\]\{\}\|]", text)
        return len(special_characters)
    if type(text) == float:
        return np.NaN

def feature_max_special_character(text):
    if type(text) == str:
        special_characters = re.findall(r"[\@\#\$\%\^\&\~\`\*\(\)\<\>\\\[\]\{\}\|]", text)
        special_char_count = Counter(special_characters)
        if special_char_count.most_common():
            max_special_char = special_char_count.most_common()[0][0]
            return max_special_char
        else:
            return "None"
    if type(text) == float:
        return np.NaN

def feature_freq_max_special_character(text):
    if type(text) == str:
        special_characters = re.findall(r"[\@\#\$\%\^\&\~\`\*\(\)\<\>\\\[\]\{\}\|]", text)
        special_char_count = Counter(special_characters)
        if special_char_count.most_common():
            freq_max_special_char = special_char_count.most_common()[0][1]
            return freq_max_special_char
        else:
            return "0"
    if type(text) == float:
        return np.NaN


# In[29]:


#Feature Extraction - Jaydeep
  
def check_single_sentence(clean_text):
    if type(clean_text) == str:
        ending_punc = re.findall(r"[.?!]", clean_text)
        ending_punc_count = Counter(ending_punc)
        single_sentence = False
        if(ending_punc_count):
            max_ep_char = max(ending_punc_count, key=ending_punc_count.get)
            max_ep_value = max(ending_punc_count.values())
        else:
            max_ep_char = ''
            max_ep_value = 0
        if max_ep_value<=1:
            single_sentence = True
    elif type(clean_text) == float:
        return np.NaN 
    return single_sentence


# In the next few cells ,we have extracted the features for the text part of all emails in the dataset using the functions defined in the above cells.

# In[30]:


email_length = df["Text"].apply(lambda row: feature_email_length_characters(row))
email_length.dropna()


# In[31]:


digit_density = df["Text"].apply(lambda row: feature_digit_density(row))
digit_density.dropna()


# In[32]:


space_density = df["Text"].apply(lambda row: feature_space_density(row))
space_density.dropna()


# In[33]:


number_paragraphs = df["Text"].apply(lambda row: feature_paragraph(row))
number_paragraphs.dropna()


# In[34]:


average_sentences_paragraph = df["Text"].apply(lambda row: feature_average_characters_paragraph(row))
average_sentences_paragraph.dropna()


# In[35]:


farewell_words = df["Text"].apply(lambda row: feature_farewell_words(row))
raw_freq_farewell_words = farewell_words.value_counts()
print(raw_freq_farewell_words)
raw_freq_farewell_words = list(raw_freq_farewell_words[raw_freq_farewell_words>20].index)
print(raw_freq_farewell_words)
freq_farewell_words = []
for word in raw_freq_farewell_words:
    tokens = nlp(word)
    for token in tokens:
        if token.pos_ not in ["PROPN"]:
            freq_farewell_words.append(token.text)

farewell_words = farewell_words.apply(lambda row: feature_freq_farewell_words(freq_farewell_words, row))
farewell_words.dropna()


# In[36]:


greeting_words = df["Text"].apply(lambda row: feature_get_greeting(row))
greeting_words.dropna()


# In[37]:


most_common_word = df["Text"].apply(lambda row: feature_most_common_word(row))
most_common_word.dropna()


# In[38]:


most_common_word[most_common_word.isna()==True]
df.iloc[1714]


# In[39]:


subjectivity = df["Text"].apply(lambda row: feature_subjectivity(row))
polarity = df["Text"].apply(lambda row: feature_polarity(row))
print(subjectivity.dropna())
print(polarity.dropna())


# In[40]:


freq_most_common_word = df["Text"].apply(lambda row: feature_freq_most_common_word(row))
freq_most_common_word.dropna()


# In[41]:


pos = df["Text"].apply(lambda row: feature_most_pos(row))
pos


# In[42]:


pos.dropna()


# In[43]:


last_punc = df["Text"].apply(lambda row: feature_ending_punctuation(row))
last_punc.dropna()


# In[44]:


freq_punc = df["Text"].apply(lambda row: feature_most_used_punctuation(row))
freq_punc.dropna()


# In[45]:


avg_len = df["Text"].apply(lambda row: avg_length(row))
avg_len


# In[46]:


avg_sent_len = df["Text"].apply(lambda row: avg_sentence_length(row))
avg_sent_len


# In[47]:


short_word_ratio = df["Text"].apply(lambda row: feature_short_word_ratio(row))
short_word_ratio.dropna()


# In[48]:


punc_freq = df["Text"].apply(lambda row: punctuation_frequency(row))
punc_freq


# In[49]:


punc_greet = df["Text"].apply(lambda row: punctuation_greeting(row))
punc_greet.dropna()


# In[50]:


number_words = df["Text"].apply(lambda row: feature_number_words(row))
number_words


# In[51]:


number_special_characters = df["Text"].apply(lambda row: feature_number_special_characters(row))
print(number_special_characters.dropna())
max_special_character = df["Text"].apply(lambda row: feature_max_special_character(row))
print(max_special_character.dropna())
freq_max_special_character = df["Text"].apply(lambda row: feature_freq_max_special_character(row))
print(freq_max_special_character.dropna())

single_sentence = df["Text"].apply(lambda row: check_single_sentence(row))
print(single_sentence.dropna())


# In[52]:


df.loc[0, ["Text", "File", "Folder"]]


# Here we are appending all the extracted features into the dataframe

# In[53]:


#Combine everything into one dataset

df["Email Length"] = email_length
df["Digit Density"] = digit_density
df["Space Density"] = space_density
df["Number of Paragraphs"] = number_paragraphs
df["Average Sentences per Paragraph"] = average_sentences_paragraph
df["Farewell Words"] = farewell_words
df["Freq Punc"] = freq_punc
df["Last Punc"] = last_punc
df["Average Word Length"] = avg_len
df["Average Sentence Length"] = avg_sent_len
df["Short Word Ratio"] = short_word_ratio
df["Punc Frequency"] = punc_freq
df["Punc after Greeting"] = punc_greet
df["Number Words"] = number_words
df["Subjectivity"] = subjectivity
df["Polarity"] = polarity
df["Most Common POS"] = pos
df["Single Sentence"] = single_sentence
df["Greeting"] = greeting_words
df["Most Common Word"] = most_common_word
df["Freq Most Common Word"] = freq_most_common_word
df["Total Special Character Count"] = number_special_characters
df["Max Occurring Special Char"] = max_special_character
df["Count of Max Special Char"] = freq_max_special_character

df


# In[54]:


df.to_csv("Enron_29_Features.csv")


# In[ ]:




