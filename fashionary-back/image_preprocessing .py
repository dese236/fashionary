#!/usr/bin/env python
# coding: utf-8

# # Fashionary - Personalized Visual Browsing
# 
# In this notebook you can find how we have created our dataset. 
# The search is powered by OpenAI's [CLIP](https://github.com/openai/CLIP) neural network.

# ## Setup Environment
# 
# In this section we will setup the environment.

# First we need to install CLIP and then make sure that we have torch 1.7.1 with CUDA support.

# In[ ]:


# !pip install git+https://github.com/openai/CLIP.git
# !pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html


# We can now load the pretrained public CLIP model.

# In[1]:


import clip
import torch

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# ## Download the Precomputed Data
# 
# In this section the precomputed feature vectors for all of the photos that we have downloaded.

# In order to compare the photos from the Amazon data to a text query, we need to compute the feature vector of each photo using CLIP.

# In[203]:


from pathlib import Path

# Set the path to the photos
photos_path = Path("/Users/ocarmeli/Downloads/AI/images_new")

# List all the photos in the folder
photos_files = list(photos_path.glob("*.png"))

# Print how many found
print(f"Photos found: {len(photos_files)}")


# In[206]:


from PIL import Image

# Function that computes the feature vectors for a batch of images
def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()


# In[210]:


import math
import numpy as np
import pandas as pd

# Define the batch size so that it fits on your GPU. You can also do the processing on the CPU, but it will be slower.
batch_size = 16

# Path where the feature vectors will be stored
features_path = Path("/Users/ocarmeli/Downloads/AI/embeddings")

# Compute how many batches are needed
batches = math.ceil(len(photos_files) / batch_size)

# Process each batch
for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    batch_ids_path = features_path / f"{i:010d}.csv"
    batch_features_path = features_path / f"{i:010d}.npy"

    # Only do the processing if the batch wasn't processed yet
    if not batch_features_path.exists():
        try:
            # Select the photos for the current batch
            batch_files = photos_files[i*batch_size : (i+1)*batch_size]

            # Compute the features and save to a numpy file
            batch_features = compute_clip_features(batch_files)
            np.save(batch_features_path, batch_features)

            # Save the photo IDs to a CSV file
            photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
            photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
            photo_ids_data.to_csv(batch_ids_path, index=False)
        except:
            # Catch problems with the processing to make the process more robust
            print(f'Problem with batch {i}')


# In[209]:


import numpy as np
import pandas as pd

# Load all numpy files
features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

# Concatenate the features and store in a merged file
features = np.concatenate(features_list)
np.save(features_path / "features_clip.npy", features)

# Load all the photo IDs
photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
photo_ids.to_csv(features_path / "photo_ids_clip.csv", index=False)


# In[ ]:


import pandas as pd
import numpy as np

# Load the photo IDs
#here you need to change path
photo_ids = pd.read_csv("/Users/ocarmeli/Downloads/AI/embeddings/photo_ids_clip.csv")
photo_ids = list(photo_ids['photo_id'])

# Load the features vectors
photo_features = np.load("/Users/ocarmeli/Downloads/AI/embeddings/features_clip.npy")

# Convert features to Tensors: Float32 on CPU and Float16 on GPU
if device == "cpu":
    photo_features = torch.from_numpy(photo_features).float().to(device)
else:
    photo_features = torch.from_numpy(photo_features).to(device)

# Print some statistics
print(f"Photos loaded: {len(photo_ids)}")


# In[ ]:


# Clean photo_ids from nan values
photo_ids_clean = []
def checkNaN(str):
    return str != str
for photo in photo_ids:
    if checkNaN(photo)==False:
        photo_ids_clean.append(photo)

