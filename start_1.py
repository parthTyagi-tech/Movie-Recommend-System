#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import ast 
import nltk 


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:



# In[4]:





# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


# In[7]:





# In[8]:


# genres
#id
#keywords
#title
#overview
#cast
#crew
movies=movies[['id','title','overview','genres','keywords','cast','crew']]


# In[9]:





# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[15]:


movies['genres']=movies['genres'].apply(convert)


# In[16]:





# In[17]:


movies.iloc[0].keywords


# In[18]:


movies['keywords']=movies['keywords'].apply(convert)


# In[19]:




# In[20]:


movies['cast'][0]


# In[21]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter=counter+1
        else:
            break
    return L


# In[22]:


movies['cast']=movies['cast'].apply(convert3)


# In[23]:





# In[24]:


movies['crew'][0]


# In[25]:


def convert4(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])

            break
    return L


# In[26]:


movies['crew']=movies['crew'].apply(convert4)


# In[27]:





# In[28]:


movies['overview'][0]


# In[29]:


movies['overview'] = movies['overview'].apply(lambda x: x.split())


# In[30]:





# In[31]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[32]:





# In[33]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[34]:


movies.head(5)


# In[35]:


movies['tags']=movies['overview'] + movies['genres'] + movies['keywords']+ movies['cast'] + movies['crew']


# In[36]:


new_df=movies[['id','title','tags']]


# In[37]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[38]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
    y=[]

    for i in text.split():
        y.append(ps.stem(i)) # convert each similar word to root word then added to list y .

    return " ".join(y) # for making each input in y as a string 


# In[39]:


new_df['tags']=new_df['tags'].apply(stem)
new_df['tags'][0]


# In[40]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[41]:


new_df['tags'][0]


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[43]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[44]:


vectors


# In[ ]:





# In[45]:


cv.get_feature_names_out()


# In[46]:


from  sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)


# In[47]:


similarity[0] # you can observe in output first movie similarity with first movies gives 1 which is logicsl 
import pickle

pickle.dump(new_df, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))


# In[48]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[49]:


def recommend(movie: str) -> list[str]:
    try:
        movie_index=new_df[new_df['title']==movie].index[0] 
    except:
        import random as rand
        movie_index = rand.randint(0, 4806)
        print("Movie not found in DB recommending random movies based on user preferences..")
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)







