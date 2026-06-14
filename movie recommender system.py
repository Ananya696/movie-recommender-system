#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head(1) ##starting rows ,,by default 5 rows


# In[4]:


credits.head()##starting rows ,,by default 5 rows


# In[5]:


credits.head(1)['cast'].values


# In[6]:


movies.merge(credits,on='title')


# In[7]:


movies.merge(credits,on='title').shape


# In[8]:


movies=movies.merge(credits,on='title')


# In[9]:


movies


# In[10]:


credits.shape


# In[11]:


movies.shape


# In[12]:


movies.shape


# In[13]:


movies.head(1)


# In[14]:


#genres
#id
#keywords
#title
#overview
#cast
#crew


# In[15]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[16]:


movies


# In[17]:


## movie_id title tags


# In[18]:


movies.head()


# In[19]:


#movie_id|title|tags......format


# In[20]:


movies.isnull().sum() ## har column me kitni rows hai jo blank hai..


# In[21]:


movies.dropna(inplace=True)##drop the movies which has any column as null


# In[22]:


movies.duplicated().sum()


# In[23]:


movies.iloc[0].genres


# In[24]:


def convert(obj):
    L=[]
    for i in obj: ##error beacuse movies aka obj is a string and i is representing its characters one by one so character ['name']  doesn't make sense 
       L.append(i['name'])
    return L


# In[25]:


import ast


# In[26]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj): ##makes obj as list so iteration is possible now and i will be dictionary one by one
        L.append(i['name'])
    return L


# In[27]:


movies['genres'] = movies['genres'].apply(convert)


# In[28]:


movies


# In[29]:


movies['keywords']


# In[30]:


movies['keywords']=movies['keywords'].apply(convert)


# In[31]:


movies


# In[32]:


movies['cast'][0]


# In[33]:


movies['cast'].isnull().sum()


# In[34]:


def convert3(obj):
    L=[]
    for i in ast.literal_eval(obj):
            L.append(i['name'])
    return L


# In[35]:


# for x in movies['cast']:
#     try:
#         convert3(x)
#     except Exception as e:
#         print("Error:",e)
#         print("Value:",x)
#         break



# In[36]:


movies['cast']=movies['cast'].apply(convert3)


# In[37]:


movies['cast']


# In[38]:


type(movies['cast'].iloc[0])


# In[39]:


movies


# In[40]:


movies['crew'][0]


# In[41]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=="Director":
            L.append(i['name'])
    return L


# In[42]:


movies['crew']=movies['crew'].apply(fetch_director)## apply() calls the function passed under it for every row one by one 


# In[43]:


movies


# In[44]:


movies['overview'][0]## a string


# In[45]:


movies['overview']=movies['overview'].apply(lambda x:x.split())## converting string into list


# In[46]:


movies


# In[47]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x]) ##to reomve spaces


# In[48]:


movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x]) ##to reomve spaces between two words


# In[49]:


movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[50]:


movies['overview']=movies['overview'].apply(lambda x:[i.replace(" ","") for i in x])


# In[51]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])


# In[52]:


movies


# In[53]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[54]:


movies


# In[55]:


new_df=movies[['movie_id','title','tags']]


# In[56]:


new_df   ## final format which we needed


# In[ ]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))  ##converting lists into strings again


# In[58]:


new_df.head(1)


# In[59]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[60]:


new_df['tags']


# In[61]:


new_df


# In[62]:


get_ipython().system('pip install nltk')


# In[63]:


pip install --upgrade pip


# In[64]:


import nltk


# In[65]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[66]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[67]:


new_df['tags']=new_df['tags'].apply(stem)


# In[68]:


get_ipython().system('pip install scikit-learn')
##installing scikit module


# In[69]:


from sklearn.feature_extraction.text import CountVectorizer  ## text to vector (vectorization)
cv = CountVectorizer(max_features=5000,stop_words='english')  ## closest vectors = similar movies 


# In[70]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[71]:


vectors


# In[72]:


feature_names=cv.get_feature_names_out() ## getting most repeated 5000 words 


# In[73]:


for i in feature_names:
    print(i)


# In[74]:


len(cv.get_feature_names_out())


# In[75]:


pip --version


# In[76]:


##higher dimension mei euclerian distance is not a good option to go with....more distance , less similarity


# In[77]:


from sklearn.metrics.pairwise import cosine_similarity #(similarity is ranging from 0 to 1)


# In[78]:


similarity=cosine_similarity(vectors) ##calculating similarity of each moview with all the movies (4806*4806)


# In[79]:


similarity


# In[80]:


list(enumerate(similarity[0]))  ##storing the indices along with the smilarity values so that similar movies can be acessed from their indices


# In[81]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[95]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances_array=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances_array)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movies_list:
       # print(new_df.iloc[i[0]].title)
       print(i)


# In[96]:


recommend("Avatar")


# In[84]:


recommend("Batman Begins")


# In[85]:


import pickle
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb')) ##converting the list into dictionary dataframe


# In[97]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[87]:


new_df


# In[88]:


new_df['title'].values


# In[89]:


new_df.to_dict()


# In[91]:


print(type(similarity))
print(similarity.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




