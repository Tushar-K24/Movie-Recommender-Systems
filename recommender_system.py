#!/usr/bin/env python
# coding: utf-8

# In[1]:

%%writefile recommender_system.py

import numpy as np
import pandas as pd
import streamlit as st


# In[2]:


movies_df=pd.read_csv('tmdb_5000_movies.csv')
credits_df=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


credits_df.rename(columns={'movie_id':'id'},inplace=True)
merged_df=movies_df.merge(credits_df, on='id')


# In[4]:


merged_df[['overview','tagline']]=merged_df[['overview','tagline']].fillna('')


# In[5]:


features=['homepage','production_countries','release_date','runtime','title_y','original_title']
merged_df.drop(features, axis=1, inplace=True)
merged_df=merged_df.rename(columns={'title_x':'title'})


# In[6]:


categories={
    "status":{"Released":0, "Post Production":1,"Rumored":2}
}
merged_df=merged_df.replace(categories)


# In[7]:


r=merged_df['vote_average']
v=merged_df['vote_count']
c=r.mean()
m=v.quantile(.90)
merged_df['weighted_rating']=(r*v + c*m)/(v+m)


# In[8]:


std_popularity = merged_df['popularity'].std()
std_rating = merged_df['weighted_rating'].std()
merged_df['popularity_norm'] = merged_df['popularity']/std_popularity
merged_df['rating_norm'] = merged_df['weighted_rating']/std_rating
merged_df['score'] = (merged_df['popularity_norm'] + merged_df['rating_norm'])/2


# In[9]:


from ast import literal_eval

features=['genres','keywords','production_companies','cast','crew']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(literal_eval)


# In[10]:


#extracting the director of the movie
def extract_director(crew):
    for i in crew:
        if i['job']=='Director':
            return i['name'];
    return np.nan #Nan if no director


# In[11]:


#extracting top 3 elements from each list
def get_top3(x):
    if isinstance(x,list):
        names=[i['name'] for i in x]
        if len(names)>3:
            return names[:3]
        return names
    return []


# In[12]:


def remove_word_spaces(x):
    if isinstance(x,list):
        return [str.lower(i.replace(' ','')) for i in x]
    else: #must come from the director
        if isinstance(x,str):
            return str.lower(x.replace(' ',''))
        return '' #no director


# In[13]:


merged_df['director']=merged_df['crew'].apply(extract_director)
features=['genres','keywords','production_companies','cast']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(get_top3)


# In[14]:


features=['genres','keywords','production_companies','cast','director']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(remove_word_spaces)


# In[15]:


def create_string(x):
    return (' '.join(x['genres']) + ' ' 
            + ' '.join(x['keywords']) + ' ' 
            + ' '.join(x['production_companies']) + ' '
            + ' '.join(x['cast']) + ' '
            + x['director']
           )
merged_df['word_string']=merged_df.apply(create_string,axis=1)


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(merged_df['word_string'])


# In[17]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[18]:


merged_df = merged_df.reset_index()
indices = pd.Series(merged_df.index, index=merged_df['title'])


# In[19]:


def get_recommendations(title, cosine_sim=cosine_sim):
    if(title!="Select an Option"):
        idx=indices[title]
        similarity=list(enumerate(cosine_sim[idx]))
        similarity.sort(key=lambda x:x[1],reverse=True)
        similarity=similarity[1:11] # first movie will be the same
        recommended_movies=[i[0] for i in similarity]
        for i in recommended_movies:
            print(merged_df['title'].iloc[i])


# In[20]:


option = st.selectbox('Select your favourite movie', merged_df['title'])
get_recommendations(option)

