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
options=np.array(merged_df['title'])
options=np.hstack((np.array(['Select an option']),options))
options=pd.Series(options)
option = st.selectbox('Select your favourite movie',options)
#option = st.selectbox('Select your favourite movie',merged_df['title'])
get_recommendations(option)

# In[ ]:
#====================================================================================================================================================
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="#F63366",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_width=px(1)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made in ",
        image('https://assets.website-files.com/5dc3b47ddc6c0c2a1af74ad0/5e181828ba9f9e92b6ebc6e7_RGB_Logomark_Color_Light_Bg.png',
              width=px(25), height=px(25)),
        " by imt-01",
        br(),
        link("https://github.com/Tushar-K24/Movie-Recommender-Systems", image('https://github.com/alooperalta/Fake-News-Detection-System/blob/main/gitLogo.png?raw=true',height="40px")),
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer()


