#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


# Here we are first iterating over all sent mails present in the dataset (all sent mails are present in
# the _sent_mail ,sent and sent_items folders). For each of the email we are extracting the sender name ,message id 
# and body part of the mail. Also ,using the text cleaning function 
# we are cleaning the body part of the mail using regex matching to remove any forwarded text that is present.

# In[2]:


def text_cleaning(raw_text):
    
    #First extract body of the email
    if re.search(r"X-FileName:.+\n+((.+\n\n?)+)", raw_text):
        text = re.search(r"X-FileName:.+\n+((?:.+\n\n?)+)", raw_text).group(1)
        
        #If the body starts with ---Forwarded return np.NaN since it does not have any useful information
        if re.match(r"-{3,} Forwarded by.+\d+\/\d+\/\d+", text):
            return np.NaN
        
        #If rest of the body of the email has forwarding in it, remove that
        if re.search(r"-{3,} Forwarded by.+\d+\/\d+\/\d+[\w\W]+", text):
            text = re.sub(r"-{3,} Forwarded by.+\d+\/\d+\/\d+[\w\W]+", "", text)
        
        #If rest of the body of the email has original message in it, remove that
        if re.search(r"-{5,}Original Message-{5,}[\w\W].*", text):
            text = re.sub(r"-{5,}Original Message-{5,}[\w\W]*[\n]*.*", "", text)
            
        if text:
            return text
        else:
            return np.NaN
        
    elif re.search(r"X-FileName:.+\n+(.+)", raw_text):
        text = re.search(r"X-FileName:.+\n+(.+)", raw_text).group(1)
        
        #If the body starts with ---Forwarded return np.NaN since it does not have any useful information
        if re.match(r"-{3,} Forwarded by.+\d+\/\d+\/\d+", text):
            return np.NaN
    
        #If rest of the body of the email has forwarding in it, remove that
        if re.search(r"-{3,} Forwarded by.+\d+\/\d+\/\d+[\w\W]+", text):
            text = re.sub(r"-{3,} Forwarded by.+\d+\/\d+\/\d+[\w\W]+", "", text)
        
        #If rest of the body of the email has original message in it, remove that
        if re.search(r"-{5,}Original Message-{5,}[\w\W].*", text):
            text = re.sub(r"-{5,}Original Message-{5,}[\w\W]*[\n]*.*", "", text)
            
        if text:
            return text
        else:
            return np.NaN
    
    else:
        return np.NaN
    
def read_email(file):
    with open(file, mode="r") as f:
        try:
            raw_text = f.read()
        except UnicodeDecodeError as e:
            return None, None, None, None
    if re.search(r"^Message-ID: <(\d+\.\d+\.)JavaMail\.evans@thyme>", raw_text):
        email_id = re.search(r"^Message-ID: <(\d+\.\d+\.)JavaMail\.evans@thyme>", raw_text).group(1)
    else:
        raise Exception("For file {} Message ID could not be found".format(file))
    if re.search(r"X-From: (.+) <?", raw_text):
        sender = re.search(r"X-From: (.+) <?", raw_text).group(1)
    else:
        print("For file {} Sender could not be found".format(file))
        sender = np.NaN
    
    text = text_cleaning(raw_text)
    
    return email_id, sender, text, raw_text

def email_extraction(author):
    root_folder = "./maildir/"
    folders = ["/_sent_mail/", "/sent/", "/sent_items/"]
    extract_data = []
    for folder in folders:
        try:
            for message in os.listdir(root_folder + author + folder):
                if os.path.isfile(root_folder + author + folder + message):
                    email_id, sender, text, raw_text = read_email(root_folder + author + folder + message)
                    if email_id != None:
                        extract_data.append([sender, author, message, email_id, text, raw_text])
        except FileNotFoundError as e:
            continue
        except NotADirectoryError as e:
            continue
    return extract_data


# Here we are calling the above functions and storing the output in a dataframe. Then we are removing duplicate emails from the dataset and any empty emails that are present

# In[3]:


#Insert path to the mail directory here
authors = os.listdir("./maildir")
number_author_folders = len(authors)
df = pd.DataFrame(columns=["Author", "Folder", "File", "Message ID", "Text", "Raw Text", "Email Folder"])

for author in authors:
    emails = email_extraction(author)
    if emails:
        df = df.append(pd.DataFrame(emails, columns=["Author", "Folder", "File", "Message ID", "Text", "Raw Text"]))

df = df.drop_duplicates(["Message ID"])
df = df[df["Text"].notna()]
# Taking only the top 20 authors
print(df.value_counts(["Folder"])[:20])
df


# Storing this data to a csv file which will be used in the later notebooks.

# In[4]:


df.to_csv("./enron.csv")


# In[5]:


df


# In[ ]:




