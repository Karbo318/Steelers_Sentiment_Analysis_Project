#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries
import pandas as pd
import numpy as np
import re,string


# In[2]:


df = pd.read_csv("Bentweets_raw.csv")


# In[3]:


#drop nonsense row
df.drop(df.index[df['user'] == 'user'][0], axis=0, inplace=True)


# In[4]:


#rename columns
df= df.rename(columns={'time': 'date', 'text': 'original_text'})


# In[5]:


#Clean up dates
df['date'][0:327]= 'Jan 14' 
df['date'] = pd.to_datetime(df['date'], format='%b %d')
df['date']= df.date.dt.strftime('%b %d')


# In[6]:


#drop null values
df = df.dropna()


# In[7]:


#cleanup user names
df['user'] = df.user.str.replace('@','')


# In[8]:


#Count duplicated
df['original_text'].duplicated().sum()


# In[9]:


df.drop_duplicates(subset= "original_text",keep="first", inplace=True)


# In[10]:


def extract_hashtags(text):
    hashtags= set(part[1:] for part in text.split() if part.startswith('#'))
    hashtags = ' '.join(hashtags)
    return hashtags

df['hashtags'] = df['original_text'].apply(lambda x: extract_hashtags(x)) 


# In[11]:


#Minimal text cleaner for (later) sentiment analysis
def min_cleaner(text):
        
    #remove links 
    text = re.sub('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', "", text)
    
    #remove words with brackets
    text = re.sub('\[.*?\]',"", text)
    
    #remove hashtags
    text = re.sub("#[A-Za-z0-9_]+","", text)

    #remove any extra spaces at beginning or end
    text= text.strip() 
    return text

df['sentiment_text'] = df['original_text'].apply(lambda x: min_cleaner(x)) 


# In[12]:


#Deep text cleaner to create a column for (later) topic modeling

def deep_cleaner(text):
    
    #make lower case
    text= text.lower()
    
     #remove links 
    text = re.sub('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', "", text)
        
    #remove hashtags
    text = re.sub("#[A-Za-z0-9_]+","", text)
    
    #remove other punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    
    #remove words with numbers
    text = re.sub('\w*\d\w*', '', text)
    
    #remove words with brackets
    text = re.sub('\[.*?\]', '', text)
    
    #remove any extra spaces at beginning or end
    text= text.strip()
    
    return text

df['topic_text'] = df['original_text'].apply(lambda x: deep_cleaner(x))


# In[13]:


clean_Bendf = df
clean_Bendf.to_pickle("clean_Bendf.pkl")

