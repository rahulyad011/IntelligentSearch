# Import libraries
import streamlit as st
import pandas as pd
from PIL import Image

# Page setup
im = Image.open("search-icon.jpeg")
st.set_page_config(page_title="Fetch Offers", page_icon=im, layout="wide")
st.image(im, width=100)
st.title("Fetch Offers")

# Connect to the Datasets
path_offers = f"data/offer_retailer.csv"
path_brand = f"data/brand_category.csv"
offers_df = pd.read_csv(path_offers, dtype=str).fillna("")
brand_df = pd.read_csv(path_brand, dtype=str).fillna("")

# Use a text_input to get the keywords to filter the data
text_search = st.text_input("Search Offers by Retailer or Brand", value="")

# Filter the dataframe using masks
m1 = offers_df["RETAILER"].str.contains(text_search)
m2 = offers_df["BRAND"].str.contains(text_search)
df_search = offers_df[m1 | m2]

# Show the filtered results
N_cards_per_row = 3
if text_search:
    for n_row, row in df_search.reset_index().iterrows():
        i = n_row%N_cards_per_row
        if i==0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # draw the card
        with cols[n_row%N_cards_per_row]:
            st.caption(f'Offer: {n_row+1}')
            if row['RETAILER']=="":
                row['RETAILER'] = "Unknown"
            if row['BRAND']=="":
                row['BRAND'] = "Unknown"
            st.markdown(f"*{row['OFFER']}*")
            st.markdown(f"Brand: **{row['BRAND'].strip()}**")
            st.markdown(f"Retailer: **{row['RETAILER'].strip()}**")
