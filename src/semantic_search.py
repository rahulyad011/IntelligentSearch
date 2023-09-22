import pandas as pd
import numpy as np
import os, sys
import pickle
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import torch

import json

import logging

logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., INFO, DEBUG, WARNING)
    format='%(asctime)s [%(levelname)s] %(message)s',  # Define log format
    filename='logs.log'  # Specify the log file name (optional)
)

config_file_path = "..\config.json"  # Change this to the path of your config file

class SemanticSearch():
    category_df = None
    brand_df = None
    embedder = None
    embedding_cache_path=None
    cross_encoder = None
    max_offer = 0

    def __init__(self) -> None:
        # Load configuration from config.json
        # Get the path to the directory containing config.json
        config_dir = os.path.dirname(__file__)

        # Construct the path to config.json
        config_file_path = os.path.join(config_dir, '..', 'config.json')

        with open(config_file_path, 'r') as config_file:
            config = json.load(config_file)

        embedding_name = config["embedding_name"]
        model_cache_folder = config["model_cache_folder"]
        self.embedding_cache_path = config["embedding_cache_path"]
        category_csv_path = config["category_csv_path"]
        brand_csv_path = config["brand_csv_path"]
        cross_encoder_model_name = config["cross_encoder_model"]
        self.max_offer = config["max_offers_visible"]

        #load biencoder model
        self.embedder = SentenceTransformer(embedding_name, cache_folder=model_cache_folder)

        #load cross encoder model
        self.cross_encoder_model = CrossEncoder(cross_encoder_model_name)

        # Load category data
        try:
            self.category_df = pd.read_csv(category_csv_path, dtype=str).fillna("")
        except FileNotFoundError:
            # Handle the case where the CSV file does not exist
            print(f"Error: The path provided for data file does not exist.")
            sys.exit(1)
        self.category_df['PRODUCT_CATEGORY'] = self.category_df['PRODUCT_CATEGORY'].str.strip().str.lower()

        # Load brand data
        try:
            self.brand_df = pd.read_csv(brand_csv_path, dtype=str).fillna("")
        except FileNotFoundError:
            # Handle the case where the CSV file does not exist
            print(f"Error: The path provided for data file does not exist.")
            sys.exit(1)
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
        try:
            if not os.path.exists(embedding_path):
                # read your corpus etc
                logging.debug("stored embedding not found")
                corpus_sentences = self.load_categories(self.category_df, 'PRODUCT_CATEGORY')
                logging.debug("Encoding the category corpus. This might take a while")
                corpus_embeddings = self.embedder.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
                corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

                logging.debug("Storing embedding file on the drive")
                with open(embedding_path, "wb") as fOut:
                    pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
            else:
                logging.debug("Loading pre-computed saved embeddings from drive")
                with open(embedding_path, "rb") as fIn:
                    cache_data = pickle.load(fIn)
                    corpus_sentences = cache_data['sentences']
                    corpus_embeddings = cache_data['embeddings']
            return corpus_sentences, corpus_embeddings
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)

    def semantic_search_offers(self, user_query, df_offers):
        data = self.load_categories(self.category_df, 'PRODUCT_CATEGORY')
        sentence_data, embeddings_saved = self.load_embeddings(data)
        categories = self.semantic_search_category(sentence_data, embeddings_saved, user_query)
        brands = self.exact_brand_match(categories)
        offers_selected = self.filter_and_rerank_offers(df_offers, brands, user_query)
        return offers_selected

    def semantic_search_category(self, sentences, embeddings, user_query):
        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(5, len(sentences))
        query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
        # we can also use util.semantic_search to perform cosine similarty + topk
        hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)
        # Retrieve the most similar categories in a dataframe
        threshold_match_category = 0.2
        matched_categories = set()
        for doc in hits[0]:
            matched = self.category_df.iloc[int(doc['corpus_id'])]['PRODUCT_CATEGORY']
            if doc['score'] >= threshold_match_category:
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

    def filter_and_rerank_offers(self, df_offers, semantic_searched_brands, input_query):
        brand_offers_all_list = []
        for brand, score in semantic_searched_brands:
            brand_offers = df_offers[df_offers['BRAND'] == brand].copy()
            if brand_offers.empty:
                continue
            brand_offers['score']=score
            brand_offers_all_list.append(brand_offers)
        if brand_offers_all_list:
            brand_offers_all = pd.concat(brand_offers_all_list, ignore_index=True)
        else:
            # empty dataframe
            return pd.DataFrame()
        # select 40 or less offers from semantic search
        brand_offers_all = brand_offers_all.iloc[:40]
        # Extract offer descriptions for the current brand
        offer_descriptions = brand_offers_all['OFFER'].tolist()
        # Create a list of tuples where the first element is the query (input_query)
        # and the second element is the offer description
        cross_inp = [(input_query, offer) for offer in offer_descriptions]
        # Score all offers for the current query with the cross_encoder
        cross_scores = self.cross_encoder_model.predict(cross_inp)
        min_score = np.min(cross_scores)
        max_score = np.max(cross_scores)
        normalized_scores = (cross_scores - min_score) / (max_score - min_score)
        weighted_scores = brand_offers_all['score']*normalized_scores
        brand_offers_all['cross-score'] = cross_scores
        brand_offers_all['score'] = np.round(weighted_scores,2)
        # Sort the offers by cross-encoder score in descending order
        brand_offers_all = brand_offers_all.sort_values(by='score', ascending=False)
        top_offers = brand_offers_all.iloc[:self.max_offer].copy()
        return top_offers