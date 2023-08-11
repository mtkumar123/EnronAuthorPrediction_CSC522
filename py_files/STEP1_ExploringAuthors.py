#!/usr/bin/env python
# coding: utf-8

# In[17]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import matplotlib.pyplot as plt


#     Extracting the number of sent emails present in the folders sent and _sent_mail for every author and storing the data in pandas dataframe

# In[18]:


#Please insert path to the mail directory here
mail_dir = "../../../../maildir/"
authors = os.listdir(mail_dir)
number_author_folders = len(authors)
extract_data =[]
for author in authors:
    try:
        number_sent_emails = len(os.listdir(mail_dir + author + "/_sent_mail"))
        extract_data.append([author, number_sent_emails, 1, 0])
    except FileNotFoundError as e:
        try:
            number_sent_emails = len(os.listdir(mail_dir + author + "/sent"))
            extract_data.append([author, number_sent_emails, 0, 1])
        except FileNotFoundError as e:
            pass
    except NotADirectoryError as e:
        pass
df = pd.DataFrame(extract_data, columns=["AuthorName", "NumberEmailsSent", "SentEmailFolder", "SentFolder"])
number_authors = df.shape[0]
number_authors


# In[19]:


print("Percentage of authors that had a sent/sent_email folder = {}".format(number_authors/number_author_folders))
print("Number of authors where we use the SentEmailFolder = {}".format(len(df.loc[df["SentEmailFolder"]==1])))
print("Number of authors where we use the SentFolder = {}".format(len(df.loc[df["SentFolder"]==1])))
print("Total number of author folders {}".format(number_author_folders))


# Sorting dataframe based on the number of sent mails

# In[20]:


df = df.sort_values(by="NumberEmailsSent", ascending=False)
# get_ipython().run_line_magic('matplotlib', 'qt')
# plt.plot(df["AuthorName"], df["NumberEmailsSent"])
# plt.show()


# In[21]:


df[:10].tail(1)
#For 1st 10 authors the lowest NumberEmailsSent is 1632


# In[22]:


df[:15].tail(1)
#For 1st 15 authors the lowest NumberEmailsSent is 1315


# In[23]:


df[:20].tail(1)
#For 1st 15 authors the lowest NumberEmailsSent is 991


# In[24]:


df[:25].tail(1)
#For 1st 25 authors the lowest NumberEmailsSent is 815


# In[25]:


df[:20]


# In[26]:


df.loc[:20, ["AuthorName"]]


# Visualizing the distribution of sent mails accross the authors.

# In[27]:


# get_ipython().run_line_magic('matplotlib', 'inline')
plt.bar(df["AuthorName"][:20], df["NumberEmailsSent"][:20])
plt.ylabel("Number of Emails")
plt.xticks(df["AuthorName"][:20], rotation="vertical")
plt.show()


# In[ ]:
