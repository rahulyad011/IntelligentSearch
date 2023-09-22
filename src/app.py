# Import libraries
import streamlit as st
import pandas as pd
from PIL import Image

from search import find_match

# Page setup
im = Image.open("search-icon.jpeg")
st.set_page_config(page_title="Fetch Offers", page_icon=im, layout="wide")
st.image(im, width=100)
st.title("Fetch Offers")

text_search = st.text_input("Search Offers by Retailer or Brand", value="")
df_search_results = find_match(text_search)

N_cards_per_row = 3
if text_search:
    for n_row, row in df_search_results.reset_index().iterrows():
        i = n_row % N_cards_per_row
        if i == 0:
            st.write("---")
            cols = st.columns(N_cards_per_row, gap="large")
        # Create a card-like layout for each offer
        with cols[i]:
            if row['RETAILER']=="":
                row['RETAILER'] = "Unknown"
            if row['BRAND']=="":
                row['BRAND'] = "Unknown"
            st.write(f"**Offer {n_row + 1}**")
            st.write(f"**<font color='lightgreen'>{row['OFFER']}</font>**", unsafe_allow_html=True)
            st.image('offer_image2.jpeg', use_column_width=True)  # Add an image for the offer
            st.write(f"**Brand:** {row['BRAND'].strip().upper()}")
            st.write(f"**Retailer:** {row['RETAILER'].strip().upper()}")
            st.write(f"**Relevance Score:** {row['score']}")
        # Add some spacing between cards
        st.write("")

st.write("---") 
st.write("Thank you for viewing the offers!")
