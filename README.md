![](Steelers_Sentiment_Analysis_Project/images/ben.png)

# Big Ben's Last Game: Sentiment Analysis of Tweets from the Steelers' Wild Card Game (January 17,2022) 
I am a huge fan of professional football and an even bigger fan of the Steelers. Knowing that it was likely to be Big Ben's last game, I saw an opportunity to use machine learning to document fan sentiment about the Steelers and Ben during their Wild Card game against the Chiefs. Using selenium I scraped over 10,000 tweets about the Steelers before and during the game. After compiling the tweets and cleaning them, I undertook a detailed word count analysis, using bar graphs, word clouds, and line graphs for visualization. Since the scraped tweets did not have sentiment labels, I trained a logistic regression model to estimate their sentiment scores using a previously labeled data set of tweets from kaggle (https://www.kaggle.com/kazanova/sentiment140/version/2), supplmented with some Steelers tweets that I manually labeled myself (so that the algorithm can gain some familiarity with them). The model achieved a .78 accuracy score, .81 and .73 precision scores on the positive and negative classes (respectively), and a macro f1 score of .77, which are all substantially better than the metrics achieved by the pre-trained Vader and TextBlob models from the nltk library in pre-trials. In the near future I plan to undertake some topic modeling and to train a deep learning model to analyse the sentiment of the tweets. Constructive feedback is always welcome!    

