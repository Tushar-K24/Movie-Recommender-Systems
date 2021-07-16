#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st


# In[2]:


filename = 'vectorizer.pkl'
pickle_in = open(filename, 'rb')
loaded_vectorizer = pickle.load(pickle_in)


# In[3]:


movies_df=pd.read_csv('./datasets/tmdb_5000_movies.csv')
credits_df=pd.read_csv('./datasets/tmdb_5000_credits.csv')


# In[4]:


credits_df.rename(columns={'movie_id':'id'},inplace=True)
merged_df=movies_df.merge(credits_df, on='id')


# In[5]:


merged_df[['overview','tagline']]=merged_df[['overview','tagline']].fillna('')


# In[6]:


features=['homepage','production_countries','release_date','runtime','title_y','original_title']
merged_df.drop(features, axis=1, inplace=True)
merged_df=merged_df.rename(columns={'title_x':'title'})


# In[7]:


categories={
    "status":{"Released":0, "Post Production":1,"Rumored":2}
}
merged_df=merged_df.replace(categories)


# In[8]:


r=merged_df['vote_average']
v=merged_df['vote_count']
c=r.mean()
m=v.quantile(.90)
merged_df['weighted_rating']=(r*v + c*m)/(v+m)


# In[9]:


std_popularity = merged_df['popularity'].std()
std_rating = merged_df['weighted_rating'].std()
merged_df['popularity_norm'] = merged_df['popularity']/std_popularity
merged_df['rating_norm'] = merged_df['weighted_rating']/std_rating
merged_df['score'] = (merged_df['popularity_norm'] + merged_df['rating_norm'])/2


# In[10]:


from ast import literal_eval

features=['genres','keywords','production_companies','cast','crew']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(literal_eval)


# In[11]:


#extracting the director of the movie
def extract_director(crew):
    for i in crew:
        if i['job']=='Director':
            return i['name'];
    return np.nan #Nan if no director


# In[12]:


#extracting top 3 elements from each list
def get_top3(x):
    if isinstance(x,list):
        names=[i['name'] for i in x]
        if len(names)>3:
            return names[:3]
        return names
    return []


# In[13]:


#extract cast
def extract_cast(x):
    if isinstance(x,list):
        male=[i['name'] for i in x if i['gender']==2]
        female=[i['name'] for i in x if i['gender']==1]
        names=[]
        if len(male)>3:
            names.extend(male[:3])
        else:
            names.extend(male)
        if len(female)>3:
            names.extend(female[:3])
        else:
            names.extend(female)
        return names
    return []


# In[14]:


def remove_word_spaces(x):
    if isinstance(x,list):
        return [str.lower(i.replace(' ','')) for i in x]
    else: #must come from the director
        if isinstance(x,str):
            return str.lower(x.replace(' ',''))
        return '' #no director


# In[15]:


#for overview and tagline
def remove_capital_letters(x):
    if isinstance(x,str):
        return str.lower(x)
    return ''


# In[16]:


merged_df['director']=merged_df['crew'].apply(extract_director)
features=['genres','keywords','production_companies']
for feature in features:
    merged_df[feature]=merged_df[feature].apply(get_top3)
merged_df['cast']=merged_df['cast'].apply(extract_cast)


# In[17]:


features=['genres','keywords','production_companies','cast','director']
for feature in features:
    merged_df[feature] = merged_df[feature].apply(remove_word_spaces)


# In[18]:


features=['overview','tagline']
for feature in features:
    merged_df[feature] = merged_df[feature].apply(remove_capital_letters)


# In[19]:


def create_string(x):
    return (' '.join(x['genres']) + ' ' 
            + ' '.join(x['cast']) + ' '
            + ' '.join(x['keywords']) + ' ' 
            + x['director'] + ' '
            + x['overview'] + ' ' 
            #+ x['tagline']
           )
merged_df['word_string']=merged_df.apply(create_string,axis=1)


# In[20]:

#count = TfidfVectorizer(stop_words='english')
count_matrix = loaded_vectorizer.fit_transform(merged_df['word_string'])


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)


# In[22]:


merged_df = merged_df.reset_index()
indices = pd.Series(merged_df.index, index=merged_df['title'])


# In[23]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx=indices[title]
    similarity=list(enumerate(cosine_sim[idx]))
    similarity.sort(key=lambda x:x[1],reverse=True)
    similarity=similarity[1:11] # first movie will be the same
    recommended_movies=[i[0] for i in similarity]
    return recommended_movies


# In[24]:

def get_recommendations(title, cosine_sim=cosine_sim):
    if title != 'Select an option':
        idx=indices[title]
        similarity=list(enumerate(cosine_sim[idx]))
        similarity.sort(key=lambda x:x[1],reverse=True)
        similarity=similarity[1:8] # first movie will be the same
        recommended_movies=[i[0] for i in similarity]
        for i in recommended_movies:
            st.write(merged_df['title'].iloc[i])

# In[25]:

option = st.selectbox('Select your favourite movie',merged_df['title'])
get_recommendations(option)


#filename = 'vectorizer.pkl'
#pickle_out = open(filename, 'wb')
#pickle.dump(count, pickle_out)
#pickle_out.close()


# In[ ]:




