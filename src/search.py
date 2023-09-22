import pandas as pd
import fuzzywuzzy
from fuzzywuzzy import fuzz

from semantic_search import SemanticSearch

import logging

logging.basicConfig(
    level=logging.INFO,  # Set the log level (e.g., INFO, DEBUG, WARNING)
    format='%(asctime)s [%(levelname)s] %(message)s',  # Define log format
    filename='logs.log'  # Specify the log file name (optional)
)

# Connect to the Datasets
path_offers = f"data/offer_retailer.csv"
path_brand = f"data/brand_category.csv"
offers_df = pd.read_csv(path_offers, dtype=str).fillna("")
brand_df = pd.read_csv(path_brand, dtype=str).fillna("")
# Clean and convert to lowercase in offer_data
offers_df['BRAND'] = offers_df['BRAND'].str.strip().str.lower()
offers_df['RETAILER'] = offers_df['RETAILER'].str.strip().str.lower()

obj_semantic_search = SemanticSearch()

def find_match(input_query):
    input_query = input_query.strip().lower()
    logging.debug("exact match with edit distance enabled")
    retailer_search = exact_match_edit_distance(input_query, offers_df, "RETAILER")
    brand_search = exact_match_edit_distance(input_query, offers_df, "BRAND")
    initial_search = pd.concat([retailer_search, brand_search], ignore_index=True, sort=False)
    if initial_search.shape[0]:
        return initial_search
    else:
        logging.debug("semantic similarity match enabled")
        semantic_searched_offers = obj_semantic_search.semantic_search_offers(input_query, offers_df)
        return semantic_searched_offers

# Function to perform exact match search with edit distance
def exact_match_edit_distance(query, df, column_name, threshold=90):
    matching_results = []
    for index, row in df.iterrows():
        text_to_match = row[column_name]
        # Calculate the edit (Levenshtein) distance
        similarity_score = fuzz.ratio(query.lower(), text_to_match.lower())
        row['score']=round(similarity_score/100, 3)
        # Check if the similarity score is above the threshold
        if similarity_score >= threshold:
            matching_results.append(row)
    return pd.DataFrame(matching_results)