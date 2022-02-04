#!/usr/bin/env python
# coding: utf-8

# In[1]:


from TwitterScraper import twitter_scraper


# In[2]:


#Use twitter scraper to get 4000 tweets about the Steelers in the hour (or so) leading up to the game 
#Start time 7:10pm EST

address = "https://twitter.com/search?q=Steelers&src=recent_search_click&f=live" 
path = '/Users/karbo/OneDrive/Desktop/sample_project_1/chromedriver'
n_tweets= 4000
scroll_size = 350
df= twitter_scraper(address,path, n_tweets,scroll_size,headless=True,csv=True)


# In[ ]:




