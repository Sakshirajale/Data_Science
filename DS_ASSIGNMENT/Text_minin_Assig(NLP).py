# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:45:44 2024

@author: ACER
"""
'''
Problem Statement: -
In the era of widespread internet use, it is necessary for businesses to understand what the consumers think of their products. 
If they can understand what the consumers like or dislike about their products, they can improve them and thereby increase their profits 
by keeping their customers happy. For this reason, they analyze the reviews of their products on websites such as Amazon or Snapdeal by 
using text mining and sentiment analysis techniques. 
'''


'''
Task 1:
1.	Extract reviews of any product from e-commerce website Amazon.
2.	Perform sentiment analysis on this extracted data and build a unigram and bigram word cloud. 
'''
from bs4 import BeautifulSoup as bs
import requests
link="https://www.amazon.com/dp/B0BV9WCGN3/ref=sspa_dk_detail_9?pf_rd_p=386c274b-4bfe-4421-9052-a1a56db557ab&pf_rd_r=WYS613S3DVV7BJ6RNVVV&pd_rd_wg=i3Nzp&pd_rd_w=0iqeN&content-id=amzn1.sym.386c274b-4bfe-4421-9052-a1a56db557ab&pd_rd_r=c1357455-5436-4537-b6d7-f517c8033d87&s=amazon-devices&sp_csd=d2lkZ2V0TmFtZT1zcF9kZXRhaWxfdGhlbWF0aWM&th=1"
page=requests.get(link)
page
page.content
## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())
############for title############
title1=soup.find_all('div',class_='a-row')
title1

review_titles1=[]
for i in range(0,len(title1)):
    review_titles1.append(title1[i].get_text())

review_titles1
review_titles1=[title1.strip('\n') for title1 in review_titles1]
review_titles1
len(review_titles1)
############for rating
rating1=soup.find_all('span',class_='a-icon-alt')
rating1
###we got the data
rate1=[]
for i in range(0,len(rating1)):
    rate1.append(rating1[i].get_text())
rate1
rate1[:]=[r.strip('\n') for r in rate1]
rate1
len(rate1)
#############Review body############
review1=soup.find_all('div',class_='a-row a-spacing-small review-data')
review1
review_body1=[]
for i in range(0,len(review1)):
    review_body1.append(review1[i].get_text())
review_body1
review_body1=[reviews1.strip('\n') for reviews1 in review_body1]
len(review_body1)
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['review_title1']=review_titles1
df['rate1']=rate1
df['review_body1']=review_body1
df
df.to_csv("C:/8-Text_mining/text_mining/Amazon01.csv",index=True)
#sentiment analysis
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
sent="This is very good Product"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("C:/8-Text_mining/text_mining/Amazon01.csv")
df.head()
df['polarity']=df['review_body1'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']


############################################################
'''
Task 2:
1.	Extract reviews for any movie from IMDB and perform sentiment analysis.
'''
import bs4
from bs4 import BeautifulSoup as bs
import requests
link="https://www.imdb.com/title/tt27510174/reviews/?ref_=tt_ov_ql_2"
page=requests.get(link)
page
page.content
## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())
#for title
title=soup.find_all('div',class_="row")
title

review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())

review_titles
review_titles=[title.strip('\n') for title in review_titles]
review_titles
len(review_titles)
##for rating
rating=soup.find_all('span',class_='rating-other-user-rating')
rating
###we got the data
rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
rate[:]=[r.strip('\n') for r in rate]
rate
len(rate)

rate2=[]
for i in range(len(rate)):
    rate2.append(int(rate[i].split('/')[0]))
rate2
##Review body
review=soup.find_all('div',class_='col')
review
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
review_body=[reviews.strip('\n') for reviews in review_body]
len(review_body)
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['review_title']=review_titles
df['rate']=rate
df['review_body']=review_body
df
df.to_csv("C:/8-Text_mining/text_mining/Movie01.csv",index=True)
#sentiment analysis
import pandas as pd
from textblob import TextBlob
sent="This is very good Movie"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("C:/8-Text_mining/text_mining/Movie01.csv")
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
#################################
'''
Task 3: 
1.Choose any other website on the internet and do some research on how to extract text and perform sentiment analysis

'''
import bs4
from bs4 import BeautifulSoup as bs
import requests
link="https://www.amazon.com/JXMOX-Charging-Compatible-Samsung-Charger/dp/B0C4GKP8KF/ref=slsr_d_dpds_fsdp4star_fa_xcat_cheapdynam_d_sccl_3_5/135-4482969-4098469?pd_rd_w=Sdhcs&content-id=amzn1.sym.189ba0e8-243a-408f-947d-77c5ac846d2e&pf_rd_p=189ba0e8-243a-408f-947d-77c5ac846d2e&pf_rd_r=6QSJ95TJ7G9EF3RZSEJT&pd_rd_wg=yjl4V&pd_rd_r=d2a9ac08-86e9-422c-8501-719f7e1572c1&pd_rd_i=B0C4GKP8KF&th=1"
page=requests.get(link)
page
page.content
## now let us parse the html page
soup=bs(page.content,'html.parser')
print(soup.prettify())
#for title
title=soup.find_all('div',class_='a-row')
title

review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())

review_titles
review_titles=[title.strip('\n') for title in review_titles]
review_titles
len(review_titles)
##for rating
rating=soup.find_all('span',class_='a-icon-alt')
rating
###we got the data
rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
rate[:]=[r.strip('\n') for r in rate]
rate
len(rate)

rate2=[]
for i in range(len(rate)):
    rate2.append(int(rate[i].split('/')[0]))
rate2
##Review body
review=soup.find_all('span',class_='a-size-base')
review
review_body=[]
for i in range(0,len(review)):
    review_body.append(review[i].get_text())
review_body
review_body=[reviews.strip('\n') for reviews in review_body]
len(review_body)
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['review_title']=review_titles
df['rate']=rate
df['review_body']=review_body
df
df.to_csv("C:/8-Text_mining/text_mining/Myntra1.csv",index=True)
#sentiment analysis
import pandas as pd
from textblob import TextBlob
sent="This is very good bag."
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv("C:/8-Text_mining/text_mining/Myntra1.csv")
df.head()
df['polarity']=df['review_title'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']

###################################################################






