import io
import clip
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import json
import math
from PIL import Image
import urllib3
import certifi
import os.path
from os import path
from IPython.core.display import HTML

max_products_in_line = 4

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Set the path to the photos
photos_path = Path("static/images_new")

# List all the photos in the folder
photos_files = list(photos_path.glob("*.png"))

df = pd.read_csv("static/df_products.csv")

# Load the photo IDs
photo_ids = pd.read_csv("static/photo_ids_clip.csv")
photo_ids = list(photo_ids['photo_id'])

#clean photo_ids from nan values
photo_ids_clean = []
def checkNaN(str):
    return str != str
for photo in photo_ids:
    if checkNaN(photo)==False:
        photo_ids_clean.append(photo)

photo_ids = photo_ids_clean

# Load the features vectors
photo_features = np.load("static/features_clip.npy")

# Convert features to Tensors: Float32 on CPU and Float16 on GPU
if device == "cpu":
    photo_features = torch.from_numpy(photo_features).float().to(device)
else:
    photo_features = torch.from_numpy(photo_features).to(device)

def encode_search_query(search_query):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    # Retrieve the feature vector
    return text_encoded

def find_best_matches(text_features, photo_features, photo_ids, results_count=3):
    # Compute the similarity between the search query and each photo using the Cosine similarity
    similarities = (photo_features @ text_features.T).squeeze(1)

    # Sort the photos by their similarity score
    best_photo_idx = (-similarities).argsort()

    # Return the photo IDs of the best matches
    return [photo_ids[i] for i in best_photo_idx[:results_count]]

def display_photo(photo_id):
    # Get the URL of the photo resized to have a width of 320px
    photo = (f"static/images_new/{photo_id}.png")
    # Display the photo
    display(Image.open(photo))

def search(search_query, photo_features, photo_ids, results_count=3):
    # in case the input is an image id
    image_flag = False
    if search_query.startswith('image'):
        text_features = encode_search_query('')
        query_photo_index = photo_ids.index(search_query)
        features = text_features*0 + photo_features[query_photo_index]
        image_flag = True
    elif search_query.startswith('http'):
        text_features = encode_search_query('')
        http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where())

        # Open the image using the URL stored in the variable 'x'
        response = http.request('GET', search_query)
        photo = Image.open(io.BytesIO(response.data))
        pre_photo = preprocess(photo).to(device).unsqueeze(0)
        cur_photo_feature = model.encode_image(pre_photo)
        cur_photo_feature /= cur_photo_feature.norm(dim=-1, keepdim=True)
        features = text_features*0 + cur_photo_feature
        image_flag = True
    #in case the input is text
    else: 
        # Encode the search query
        features = encode_search_query(search_query)

    best_photo_ids = find_best_matches(features, photo_features, photo_ids, 4)
  
    # Find the best matches
    photos = []
    brands = []
    styles = []

    for photo_id in best_photo_ids:
        photos.append({"id": photo_id, "url" : get_image_url(photo_id) , "brand" : get_brand(photo_id) , "style" : get_style(photo_id) , "rate" : get_rate(photo_id) })
        brands.append(get_brand(photo_id))
        styles.append(get_style(photo_id))
    brands_dict= []
    for brand in np.unique(np.array(brands)).tolist():
            brands_dict.append({brand : brand})

    return {"best_photo_ids" : photos ,"brands":brands_dict, "styles":styles,  'headers':{"Access-Control-Allow-Origin": "*"} } 

def get_images(search_query): 
    return search(search_query, photo_features, photo_ids_clean , 4)

#Util functions for recieving image features from data based on image ID
def get_image_url(id):
    id_split = id.split('_')
    return df[(df.asin == str(id_split[1]))].imageURLHighRes.values[0].split("'")[1::2][int(id_split[2])]

def get_brand(id):
    id_split = id.split('_')
    brand = df[(df.asin == str(id_split[1]))].brand.values[0]
    if checkNaN(brand) :
        return ""
    else:
        return brand  

def get_rate(id):
    id_split = id.split('_')
    return round(df[(df.asin == str(id_split[1]))].average_rating).values[0]

def get_style(id):
    id_split = id.split('_')
    style_str = df[(df.asin == str(id_split[1]))]["style"].values[0]
    return json.loads(style_str.replace("'" , '"'))
   
def see_similar(photo_id):
    photo_id_num = photo_id[-1]
    photos = []
    for i in range(max_products_in_line):
        if i!= photo_id_num and path.exists(f"static/images_new/{photo_id[:-1]}{i}.png"):
            photos.append(photo_id[:-1]+f'{i}')
    return {"photos" : photos ,  'headers':{"Access-Control-Allow-Origin": "*"}}

def search_and_concept(search_query, photo_features, photo_ids, concept, photo_weight=0.5 , results_count=4):
    # Encode the search query
    image_flag = False
    text_features = encode_search_query(concept)
    if search_query.startswith('image'):
        # Find the feature vector for the specified photo ID
        query_photo_index = photo_ids.index(search_query)
        query_photo_features = photo_features[query_photo_index]

        # Combine the test and photo queries and normalize again
        search_features = text_features + query_photo_features * photo_weight
        search_features /= search_features.norm(dim=-1, keepdim=True)

        # Find the best match
        best_photo_ids = find_best_matches(search_features, photo_features, photo_ids, results_count)
        image_flag = True
        
    elif search_query.startswith('http'):
        http = urllib3.PoolManager(
        cert_reqs='CERT_REQUIRED',
        ca_certs=certifi.where()
            )
        # Open the image using the URL stored in the variable 'x'
        response = http.request('GET', search_query)
        photo = Image.open(io.BytesIO(response.data))
        pre_photo = preprocess(photo).to(device).unsqueeze(0)
        cur_photo_feature = model.encode_image(pre_photo)
        cur_photo_feature /= cur_photo_feature.norm(dim=-1, keepdim=True)
        search_features = text_features + cur_photo_feature * photo_weight
        best_photo_ids = find_best_matches(search_features, photo_features, photo_ids, results_count)
        image_flag = True

    else:
        search_features = encode_search_query(search_query+ ' ' + concept)
        best_photo_ids = find_best_matches(search_features, photo_features, photo_ids, results_count)
        # Find the best matches
    photos = []
    for photo_id in best_photo_ids:
        photos.append({"id": photo_id, "url" : get_image_url(photo_id)})

    return {"best_photo_ids" : photos , 'headers':{"Access-Control-Allow-Origin": "*"} } 

def get_concept(search_query , concept):
    return search_and_concept(search_query, photo_features, photo_ids, concept,photo_weight=0.5 , results_count=4)
