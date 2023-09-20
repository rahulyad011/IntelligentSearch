import pandas as pd
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer, util
import torch

import json

# from search import exact_match_edit_distance

config_file = "config.json"  # Change this to the path of your config file

class SemanticSearch():
    category_df = None
    brand_df = None
    embedder = None
    def __init__(self) -> None:
        config = self.load_config(config_file)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models')
        path_category = f"data/categories.csv"
        self.category_df = pd.read_csv(path_category, dtype=str).fillna("")
        self.category_df['PRODUCT_CATEGORY'] = self.category_df['PRODUCT_CATEGORY'].str.strip().str.lower()
        path_brand = f"data/brand_category.csv"
        self.brand_df = pd.read_csv(path_brand, dtype=str).fillna("")
        self.brand_df['BRAND'] = self.brand_df['BRAND'].str.strip().str.lower()
        self.brand_df = self.brand_df.rename(columns={"BRAND_BELONGS_TO_CATEGORY": "PRODUCT_CATEGORY"})
        self.brand_df['PRODUCT_CATEGORY'] = self.brand_df['PRODUCT_CATEGORY'].str.strip().str.lower()

    # Function to read the configuration from a JSON file
    def load_config(self, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
        return config

    def load_categories(self, df, target_col):
        category_data = []
        for category in df['PRODUCT_CATEGORY']:
            # category_data.append([category['PRODUCT_CATEGORY'], category['IS_CHILD_CATEGORY_TO']])
            category_data.append(category)
        return category_data

    def load_embeddings(self, data):
        corpus_embeddings = self.embedder.encode(data, convert_to_tensor=True)
        embedding_cache_path='models/embeddings.pt'
        if not os.path.exists(embedding_cache_path):
            # read your corpus etc
            corpus_sentences = self.load_data(self.category_df)
            print("Encoding the corpus. This might take a while")
            corpus_embeddings = self.embedder.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
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

    def semantic_search_offers(self,user_query):
        data = self.load_categories(self.category_df, 'PRODUCT_CATEGORY')
        sentence_data, embeddings_saved = self.load_embeddings(data)
        categories = self.semantic_search_category(sentence_data, embeddings_saved, user_query)
        brands = self.exact_brand_match(categories)
        return brands

    def semantic_search_category(self, sentences, embeddings, user_query):
        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(5, len(sentences))
        query_embedding = self.embedder.encode(user_query, convert_to_tensor=True)
        # We use cosine-similarity and torch.topk to find the highest 5 scores
        # cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        # top_results = torch.topk(cos_scores, k=top_k)
        # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
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
                # print("Matched doc:", doc)
                matched_categories.add((matched, doc['score']))
        # result_category = self.dataframe_utils(hits[0], threshold_match_category, self.category_df)
        return list(matched_categories)

    # Function to perform semantic search for brands in brand table
    def exact_brand_match(self, categories):
        # Join the dataframes on 'Product_Category'
        merged_data = pd.merge(self.brand_df, self.category_df, on='PRODUCT_CATEGORY', how='inner')
        # Group by 'Product_Category' and get all unique 'Brand_Name' values for each category
        category_brand_mapping = merged_data.groupby('PRODUCT_CATEGORY')['BRAND'].unique().reset_index()
        # print("category_brand_mapping")
        # print(category_brand_mapping.head())
        matched_brands = set()
        for item in categories:
            # item here as both category and the similarity score of this category with input query
            brands_per_category = category_brand_mapping.loc[category_brand_mapping['PRODUCT_CATEGORY']==item[0]]
            brand_list = list(brands_per_category['BRAND'])
            # print("brand list per category:")
            # print(brand_list)
            for b in brand_list[0]:
                # adding item[1] similairty score
                matched_brands.add((str(b), item[1]))
        return list(matched_brands)
        
    # def dataframe_utils(self, documents, threshold, target_df):
    #     # print("number of matches:", len(documents))
    #     matched_categories = set()
    #     for doc in documents:
    #         matched = target_df.iloc[int(doc['corpus_id'])]['PRODUCT_CATEGORY']
    #         if doc['score'] >= threshold:
    #             # print("Matched doc:", doc)
    #             matched_categories.add(matched)
    #     return matched_categories

# obj = SemanticSearch()
# data = obj.load_data(obj.category_df)
# dataset, embedding = obj.load_embeddings(data)
# user_input = "sauce"
# documents = obj.sim_search(dataset, embedding, user_input)
# print(documents)