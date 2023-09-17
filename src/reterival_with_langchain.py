from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import pandas as pd

embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
path_category = f"data/categories.csv"
category_df = pd.read_csv(path_category, dtype=str).fillna("")

def load_data(df):
    loader = DataFrameLoader(df, page_content_column="PRODUCT_CATEGORY")
    data = loader.load()
    db = Chroma.from_documents(data, embedding)
    return db

def semantic_search(db):
    query = "Sauce"
    docs = db.similarity_search_with_score(query)
    return docs

vectordb = load_data(category_df)
documents = semantic_search(vectordb)
print(documents)
