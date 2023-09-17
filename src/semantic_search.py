import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

import json

config_file = "config.json"  # Change this to the path of your config file

# Function to read the configuration from a JSON file
def load_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config

config = load_config(config_file)

embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models')
path_category = f"data/categories.csv"
category_df = pd.read_csv(path_category, dtype=str).fillna("")

def load_data(df):
    category_data = []
    for category in df['PRODUCT_CATEGORY']:
        # category_data.append([category['PRODUCT_CATEGORY'], category['IS_CHILD_CATEGORY_TO']])
        category_data.append(category)
    return category_data

def load_embeddings(data):
    corpus_embeddings = embedder.encode(data, convert_to_tensor=True)
    embedding_cache_path='models/embeddings.pt'
    if not os.path.exists(embedding_cache_path):
        # read your corpus etc
        corpus_sentences = load_data(category_df)
        print("Encoding the corpus. This might take a while")
        corpus_embeddings = embedder.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        print("Storing file on disc")
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
    else:
        print("Loading pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            corpus_sentences = cache_data['sentences']
            corpus_embeddings = cache_data['embeddings']
    return corpus_sentences, corpus_embeddings

def sim_search(sentences, embeddings, user_query):
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(sentences))
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    # We use cosine-similarity and torch.topk to find the highest 5 scores
    # cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    # top_results = torch.topk(cos_scores, k=top_k)
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, embeddings, top_k=5)
    for hit in hits[0]:
        print(hit)
        print(sentences[int(hit['corpus_id'])], "(Score: {:.4f})".format(hit['score']))
    return hits

data = load_data(category_df)
dataset, embedding = load_embeddings(data)
user_input = "sauce"
documents = sim_search(dataset, embedding, user_input)
# print(documents)
