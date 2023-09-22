import pandas as pd
import fuzzywuzzy
from fuzzywuzzy import fuzz

from semantic_search import SemanticSearch

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
    retailer_search = exact_match_edit_distance(input_query, offers_df, "RETAILER")
    brand_search = exact_match_edit_distance(input_query, offers_df, "BRAND")
    initial_search = pd.concat([retailer_search, brand_search], ignore_index=True, sort=False)
    print("initial_search:", initial_search, initial_search.shape)
    if initial_search.shape[0]:
        return initial_search
    else:
        semantic_searched_brands = obj_semantic_search.semantic_search_offers(input_query)
        # Create an empty DataFrame with the desired columns
        filtered_offers = pd.DataFrame(columns=['OFFER', 'RETAILER', 'BRAND', 'score'])
        # Iterate through the semantic_searched_brands and offers_df to append rows
        for brand, score in semantic_searched_brands:
            brand_offers = offers_df[offers_df['BRAND'] == brand]
            for _, row in brand_offers.iterrows():
                # Copy the row to avoid modifying the original DataFrame
                offer_details = row.copy()
                # Set the 'score' value for this row
                offer_details['score'] = round(score,3)
                # Append the modified row to the filtered_offers DataFrame
                filtered_offers = pd.concat([filtered_offers, offer_details.to_frame().T], ignore_index=True)
        # Now, filtered_offers contains the 'score' column with values set correctly
        return filtered_offers

# Function to perform exact match search with edit distance
def exact_match_edit_distance(query, df, column_name, threshold=90):
    # Create an empty list to store matching results
    matching_results = []
    # Iterate through the rows of the dataframe
    for index, row in df.iterrows():
        text_to_match = row[column_name]
        # Calculate the edit (Levenshtein) distance
        similarity_score = fuzz.ratio(query.lower(), text_to_match.lower())
        row['score']=round(similarity_score/100, 3)
        # Check if the similarity score is above the threshold
        if similarity_score >= threshold:
            matching_results.append(row)
    return pd.DataFrame(matching_results)