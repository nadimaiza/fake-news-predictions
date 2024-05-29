#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import re # helps find similar words 
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords # removes useless words such as "ok, but.."
from nltk.stem import PorterStemmer # simplify words 
from sklearn.feature_extraction.text import TfidfVectorizer # transforms words into numbers 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


import nltk
nltk.download("stopwords")


# In[4]:


print(stopwords.words("english")) # these are the words with negligeable value 


# In[5]:


data = pd.read_csv("fake news_training.csv")
df = pd.DataFrame(data)


# # DATA PREPROCESSING

# In[6]:


df.shape


# In[7]:


df.head(6)


# 0 = REAL NEWS
# 1 = FAKE NEWS

# In[8]:


df.groupby("label")["id"].count()
#we can see that we have 10387 real news out of 20800 articles


# In[9]:


df.isnull().sum()


# In[10]:


#replacing the null values with empty strings
df = df.fillna('')
df.isnull().sum()


# In[11]:


#merging author name and title
df["content"] = df["author"] + df["title"]


# In[12]:


X = df.drop(["label"], axis = 1)
y = df["label"]


# # TIME SO STEM (reducing words to their root form):

# explanatory of def stemming:
# 
# 
# re.sub(substitute) -->  re.sub(pattern, replacement, text)
# 
#                         #pattern = here we are searching for all the patterns that are not going from a-z and A-Z (so numbers)
#                         
#                         #replacement: This is the string that will replace the matched pattern. (will replace numbers and everything other than a-z and A-Z)
#                         
#                         #text: This is the original text where you want to perform the substitution.
#                         
#                         
# So, the overall effect of this line is to substitute any character that is not an uppercase or lowercase letter followed by a space with just a space. This is often used for text cleaning, where you want to remove characters that are not letters and replace them with spaces.

# In[13]:


port_stem = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub("[^a-zA-Z] "," ", content) # ^ is used to reject
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words("english")]

    # so we are going to port_stem the words 
    #on the content that we already cleaned (removed numbers when we used re.sub blablabla)
    # and if not word in stopword because we cannot and don't want to port_stem useless words (reviews the words at the top, the stopwords)
    
    stemmed_content = " ".join(stemmed_content)
    return stemmed_content


# In[14]:


df["content"] = df["content"].apply(stemming)


# In[15]:


df.head()


# SEPARATE THE DATA

# In[16]:


X = df["content"]
y = df["label"]


# # Converting textual data into numerical data

# In[17]:


vectoriser = TfidfVectorizer()
#counts the number of time a certain word has been repeated and 
#change it into numbers so that the machine can understand it 
#for the machine, the more the word is repeated, the more important it is
vectoriser.fit(X) #we don't need to do it with y because y is already a number
X = vectoriser.transform(X)


# SPLITTING THE DATA INTO TRAINING AND TESTING DATA

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.8, stratify= y, random_state= 42)


# TRAINING THE MODEL

# In[19]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
X_train_pred = lr.predict(X_train)
train_data_accuracy = accuracy_score(X_train_pred, y_train)
train_data_accuracy


# In[20]:


X_test_pred = lr.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, y_test)
test_data_accuracy


# TESTING AND MAKING SOME PREDICTIONS:

# In[21]:


X_news = X_test[3] #we are trying the model irl
real_prediction = lr.predict(X_news)
print(real_prediction)
if real_prediction == 0:
    print("The news is reliable")
else:
    print("The news is probably fake")


# In[ ]:





# In[ ]:




