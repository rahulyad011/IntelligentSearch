import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

import json

config_file_path = "..\config.json"  # Change this to the path of your config file

class SemanticSearch():
    category_df = None
    brand_df = None
    embedder = None
    embedding_cache_path=None
    # def __init__(self) -> None:
    #     config = self.load_config(config_file)
    #     self.embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models')
    #     path_category = f"data/categories.csv"
    #     self.category_df = pd.read_csv(path_category, dtype=str).fillna("")
    #     self.category_df['PRODUCT_CATEGORY'] = self.category_df['PRODUCT_CATEGORY'].str.strip().str.lower()
    #     path_brand = f"data/brand_category.csv"
    #     self.brand_df = pd.read_csv(path_brand, dtype=str).fillna("")
    #     self.brand_df['BRAND'] = self.brand_df['BRAND'].str.strip().str.lower()
    #     self.brand_df = self.brand_df.rename(columns={"BRAND_BELONGS_TO_CATEGORY": "PRODUCT_CATEGORY"})
    #     self.brand_df['PRODUCT_CATEGORY'] = self.brand_df['PRODUCT_CATEGORY'].str.strip().str.lower()

    def __init__(self) -> None:
        # Load configuration from config.json
        # Get the path to the directory containing config.json
        config_dir = os.path.dirname(__file__)  # Assuming this code is in your_code.py

        # Construct the path to config.json
        config_file_path = os.path.join(config_dir, '..', 'config.json')

        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)

        embedding_name = config["embedding_name"]
        model_cache_folder = config["model_cache_folder"]
        self.embedding_cache_path = config["embedding_cache_path"]
        category_csv_path = config["category_csv_path"]
        brand_csv_path = config["brand_csv_path"]

        self.embedder = SentenceTransformer(embedding_name, cache_folder=model_cache_folder)

        # Load category data
        self.category_df = pd.read_csv(category_csv_path, dtype=str).fillna("")
        self.category_df['PRODUCT_CATEGORY'] = self.category_df['PRODUCT_CATEGORY'].str.strip().str.lower()

        # Load brand data
        self.brand_df = pd.read_csv(brand_csv_path, dtype=str).fillna("")
        self.brand_df['BRAND'] = self.brand_df['BRAND'].str.strip().str.lower()
        self.brand_df = self.brand_df.rename(columns={'BRAND_BELONGS_TO_CATEGORY': 'PRODUCT_CATEGORY'})
        self.brand_df['PRODUCT_CATEGORY'] = self.brand_df['PRODUCT_CATEGORY'].str.strip().str.lower()

    # Function to read the configuration from a JSON file
    def load_config(self, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
        return config

    def load_categories(self, df, target_col):
        category_data = []
        for category in df[target_col]:
            category_data.append(category)
        return category_data

    def load_embeddings(self, data):
        corpus_embeddings = self.embedder.encode(data, convert_to_tensor=True)
        #load embedding file if exist
        script_dir = os.path.dirname(os.path.abspath(__file__))
        embedding_path=os.path.join(script_dir, '..', self.embedding_cache_path)
        if not os.path.exists(embedding_path):
            # read your corpus etc
            corpus_sentences = self.load_categories(self.category_df, 'PRODUCT_CATEGORY')
            print("Encoding the corpus. This might take a while")
            corpus_embeddings = self.embedder.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

            print("Storing file on disc")
            with open(embedding_path, "wb") as fOut:
                pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
        else:
            print("Loading pre-computed embeddings from disc")
            with open(embedding_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_sentences = cache_data['sentences']
                corpus_embeddings = cache_data['embeddings']
        return corpus_sentences, corpus_embeddings

    def semantic_search_offers(self,user_query):
        data = self.load_categories(self.category_df, 'PRODUCT_CATEGORY')
        sentence_data, embeddings_saved = self.load_embeddings(data)
        categories = self.semantic_search_category(sentence_data, embeddings_saved, user_query)
        print("categories: ", categories)
        brands = self.exact_brand_match(categories)
        return brands

    def semantic_search_category(self, sentences, embeddings, user_query):
        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(5, len(sentences))
        query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
        # we can also use util.semantic_search to perform cosine similarty + topk
        hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)
        for hit in hits[0]:
            print(hit)
            print(self.category_df.iloc[int(hit['corpus_id'])], "(Score: {:.4f})".format(hit['score']))
        # Get the index of the most similar category
        # Retrieve the most similar categories in a dataframe
        threshold_match_category = 0.5
        matched_categories = set()
        for doc in hits[0]:
            matched = self.category_df.iloc[int(doc['corpus_id'])]['PRODUCT_CATEGORY']
            if doc['score'] >= threshold_match_category:
                print("selected category: ", matched)
                matched_categories.add((matched, doc['score']))
        matched_categories_list = list(matched_categories)
        # Sort the matched categories by doc['score'] in descending order
        sorted_categories = sorted(matched_categories_list, key=lambda x: x[1], reverse=True)
        return sorted_categories

    # Function to perform semantic search for brands in brand table
    def exact_brand_match(self, categories):
        # Join the dataframes on 'Product_Category'
        merged_data = pd.merge(self.brand_df, self.category_df, on='PRODUCT_CATEGORY', how='inner')
        # Group by 'Product_Category' and get all unique 'Brand_Name' values for each category
        category_brand_mapping = merged_data.groupby('PRODUCT_CATEGORY')['BRAND'].unique().reset_index()
        matched_brands = set()
        for item in categories:
            # item here as both category and the similarity score of this category with input query
            brands_per_category = category_brand_mapping.loc[category_brand_mapping['PRODUCT_CATEGORY']==item[0]]
            brand_list = list(brands_per_category['BRAND'])
            for b in brand_list[0]:
                # adding item[1] similairty score
                matched_brands.add((str(b), item[1]))
        matched_brands_list =  list(matched_brands)
        # Sort the matched categories by doc['score'] in descending order
        sorted_brands = sorted(matched_brands_list, key=lambda x: x[1], reverse=True)
        return sorted_brands