#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import gzip
import pandas as pd
import urllib


# ## Data reading and filtering

# In[2]:


### load the meta data
def load_meta():
    meta_data = []
    with gzip.open('meta_AMAZON_FASHION.json.gz') as f:
        for l in f:
            meta_data.append(json.loads(l.strip()))
        
    # total length of list, this number equals total number of products
    # print(len(meta_data))

    # first row of the list
    # print(meta_data[0])
    return meta_data


# In[3]:


### load the reviews data
def load_data():
    data = []
    with gzip.open('AMAZON_FASHION.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))
        
    # total length of list, this number equals total number of products
    # print(len(data))

    # first row of the list
    # print(data[0])
    return data


# In[4]:

def merge_data(data , meta_data):
# convert list into pandas dataframe
    df_reviews = pd.DataFrame.from_dict(data)
    df_meta = pd.DataFrame.from_dict(meta_data)

    # print('df_reviews: ', len(df_reviews), ' df_meta: ', len(df_meta))


    # In[5]:


    # merging the data by amazon id
    df_merge = df_reviews.merge(df_meta, on='asin', how='left')
    df_merge.shape
    return df_merge


# In[6]:

def filter_data(df_merge):
# filtering by values we want to have
    df_merge_filtered = df_merge[(df_merge.imageURLHighRes.notnull())&(df_merge['style'].notnull())]
    df_merge_filtered.shape


    # In[ ]:


    #over this data we can create the mean rating (overall) vakue for each item
    df_merge_filtered.reset_index(drop=True, inplace=True)


    # In[17]:


    df_merge_filtered.columns
    return df_merge_filtered

def get_final_data():
# In[10]:
    # getting the first row of each item to download the photos only once
    meta_data = load_meta()
    data = load_data()
    df_merge = merge_data(data , meta_data)
    df_merge_filtered =  filter_data(df_merge)
    df_final = df_merge_filtered.groupby('asin').first().reset_index()
    df_merge_filtered.to_csv('merge_filtered.csv', index=True)
    df_final.to_csv('final.csv', index=True)
    
    return df_final , df_merge_filtered


# In[46]:


# df_merge_filtered[(df_merge_filtered.asin=='B0000AOE9U')]['price']
# df_merge_filtered = pd.read_csv('static/merge_filtered.csv')
# df_final = pd.read_csv('static/final.csv')
# df_final , df_merge_filtered = get_final_data()
# In[14]:

def get_image_urls(photo_id):
    # print(df_final[0])
    image_URLs = df_final[(df_final.asin==photo_id)]['imageURLHighRes'].tolist()
    print("image_URLs : " , df_final.columns)
    return image_URLs
# import requests
# import os.path
# from os import path

# # downloading the photos to compute the embeddings
# # count the number of images that failed to be downloaded
# count = 0
# for i in range(len(df_final)):
#     cur_images = df_final.imageURLHighRes[i]
#     for j in range(len(cur_images)):
#         url = df_final.imageURLHighRes[i][j]
#         img_id = df_final.asin[i]
#         try:
#             response = requests.get(url, verify=False)
#             with open(f'/Users/ocarmeli/Downloads/AI/images_new/image_{img_id}_{j}.png', 'wb') as f:
#                 f.write(response.content)
#         except:
#             count+=1
#             print('problem with image id: ', img_id, 'the Occurrence: ', j)
# print("done, number of problems: ", count)


# In[ ]:




