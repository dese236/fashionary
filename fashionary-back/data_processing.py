import os
import json
import gzip
import pandas as pd
import urllib

### Data reading and filtering

### load the meta data
def load_meta():
    meta_data = []
    with gzip.open('meta_AMAZON_FASHION.json.gz') as f:
        for l in f:
            meta_data.append(json.loads(l.strip()))
    return meta_data

### load the reviews data
def load_data():
    data = []
    with gzip.open('AMAZON_FASHION.json.gz') as f:
        for l in f:
            data.append(json.loads(l.strip()))
    return data

def merge_data(data , meta_data):
# convert list into pandas dataframe
    df_reviews = pd.DataFrame.from_dict(data)
    df_meta = pd.DataFrame.from_dict(meta_data)
    df_merge = df_reviews.merge(df_meta, on='asin', how='left')
    df_merge.shape
    return df_merge

def filter_data(df_merge):
# filtering by values we want to have
    df_merge_filtered = df_merge[(df_merge.imageURLHighRes.notnull())&(df_merge['style'].notnull())]
    df_merge_filtered.shape
    df_merge_filtered.reset_index(drop=True, inplace=True)
    df_merge_filtered.columns
    return df_merge_filtered

def get_final_data():
    # getting the first row of each item to download the photos only once
    meta_data = load_meta()
    data = load_data()
    df_merge = merge_data(data , meta_data)
    df_merge_filtered =  filter_data(df_merge)
    df_final = df_merge_filtered.groupby('asin').first().reset_index()
    df_merge_filtered.to_csv('merge_filtered.csv', index=True)
    df_final.to_csv('final.csv', index=True)  
    return df_final , df_merge_filtered

def get_image_urls(photo_id):
    df_final,_ = get_final_data()
    image_URLs = df_final[(df_final.asin==photo_id)]['imageURLHighRes'].tolist()
    return image_URLs





