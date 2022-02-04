#!/usr/bin/env python
# coding: utf-8

# In[1]:


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd


# In[2]:


def twitter_scraper(address,path, n_tweets, scroll_size, headless=False, csv=False):

    #initialize webdriver
    options = webdriver.ChromeOptions()

    #decide upon headless mode
    if headless == True:
        options.add_argument('headless')

    #indicate path where driver is located on computer 
    driver = webdriver.Chrome(executable_path=path, options=options)

    #feed in the website
    driver.get(address)

    #if not using headless mode maximize window size for optimal scraping
    if headless == False:
        driver.maximize_window()

    #function for extracting user, text, and time data
    def get_tweet(element):
        try:
            user = element.find_element_by_xpath(".//span[contains(text(), '@')]").text
            text = element.find_element_by_xpath(".//div[@lang]").text
            time = element.find_element_by_xpath(".//time[@datetime]").text
            tweet_data = [user, text,time]
        except:
            tweet_data = ['user', 'text','time']
        return tweet_data


    user_data = []
    text_data = []
    time_data = []
    tweet_ids = set()

    while len(tweet_ids)<n_tweets:

        #Scrape the page
        try:
            tweets = WebDriverWait(driver, 5).until(
            EC.presence_of_all_elements_located((By.XPATH, "//article[@role='article']")))

            for tweet in tweets:
                tweet_list = get_tweet(tweet)
                tweet_id = ''.join(tweet_list)

                if tweet_id not in tweet_ids:
                    tweet_ids.add(tweet_id)
                    user_data.append(tweet_list[0])
                    text_data.append(" ".join(tweet_list[1].split()))
                    time_data.append(tweet_list[2])
                    print(f'Scraping: {len(tweet_ids)} / {n_tweets}.')

                if len(tweet_ids) >= n_tweets:
                    break 

            driver.execute_script(f"window.scrollBy(0,{scroll_size})", "") #Scroll down by chosen length

        #Scroll up if any problems
        except:
            print('Re-try')
            driver.find_element_by_tag_name("body").send_keys(Keys.UP) 
            time.sleep(1)

    driver.quit()

    #Create a data frame with scraped user, text, and time data
    df = pd.DataFrame({'user': user_data, 'text': text_data, 'time': time_data})

    #if desired, save df to csv

    if csv == True: 
        df.to_csv('Secondhalftweets_raw.csv', index=False)

    return df   

