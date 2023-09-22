# Welcome to Intelligent Search

## Introduction

Welcome to the Fetch Offers Search Engine project! This intelligent search engine is designed to help users effortlessly discover and access a wide range of offers from various retailers and brands. Whether you're searching for deals by brand, retailer, or product category, this tool empowers you to find the most relevant and enticing offers.

Our search engine leverages a hybrid approach, combining precise keyword matching (with typo correction) for retailer and brand searches and advanced semantic search techniques for category-based queries. The result? A streamlined and efficient way to access the offers that matter most to you.

In this README, you'll find instructions on how to set up and run the application locally, along with a link to explore our live demo. Let's get started and uncover the best offers with ease!

**Plus, we've broken down our solution approach at the end to provide insights into how our search engine achieves precise and efficient offer retrieval.**

# Instructions for Setup

To run the Fetch Offers Search Engine tool locally, follow these steps:

1. **Clone the Repository**: Start by cloning this GitHub repository to your local machine using the following command:

   ```bash
   git clone https://github.com/rahulyad011/IntelligentSearch.git
   ```

2. **Navigate to the Project Directory**: Change your working directory to the project folder:

   ```bash
   cd IntelligentSearch
   ```

3. **Create a Virtual Environment**: It's a good practice to run the application in a virtual environment to isolate dependencies. Create a virtual environment named "venv":

   ```bash
   python -m venv venv
   ```

4. **Activate the Virtual Environment**: Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

5. **Install Dependencies**: Use `pip` to install the required Python dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

6. **Set Up Configuration**: Create a `config.json` file in the project root directory and configure it according to your requirements. Example `config.json`:

    ***Configuration Parameters***

    - `"embedding_name"`: Name of the pre-trained NLP embedding model.
    - `"embedding_cache_path"`: File path to store pre-computed embeddings.
    - `"model_cache_folder"`: Folder for caching various models.
    - `"category_csv_path"`: File path to the CSV with product category data.
    - `"brand_csv_path"`: File path to the CSV with brand and category associations.
    - `"offers_csv_path"`: File path to the CSV with offer data.
    - `"cross_encoder_model"`: Name or path of a pre-trained cross-encoder model.
    - `"max_offers_visible"`: Maximum number of displayed offers.
    
    <br>

    ***Updating Data and Recomputing Embeddings***

    If you need to update your data or refresh the embeddings used by the Fetch Offers Search Engine, follow these steps:

    1. **Delete Previous Embeddings**:
    - Locate and delete the previous embedding files.
    - These files are typically stored in the location specified by `"embedding_cache_path"` in your config.json (e.g., "models/embeddings.pt").

    2. **Recompute Embeddings**:
    - Run the code or script responsible for generating embeddings.
    - If the code doesn't find precomputed embeddings in the specified location, it will automatically compute new embeddings from updated data.

    These steps ensure that your embeddings remain up-to-date and accurate, reflecting any changes made to your data.

   ```json
   {
       "embedding_name": "all-MiniLM-L6-v2",
       "embedding_cache_path" : "models/embeddings.pt",
       "model_cache_folder": "models",
       "category_csv_path": "data/categories.csv",
       "brand_csv_path": "data/brand_category.csv",
       "offers_csv_path": "data/offer_retailer.csv",
       "cross_encoder_model":"cross-encoder/ms-marco-MiniLM-L-6-v2",
       "max_offers_visible":12
   }
   ```
   Customize the paths and settings to match your project configuration.

7. **Run the Application**: Execute the `startup.sh` bash script to start the application with Streamlit:

   ```bash
   bash startup.sh
   ```

   or
   ```bash
    ./startup.sh
   ```

8. **Access the Application**: Once the application is running, open a web browser and go to `http://localhost:8501` to access the Fetch Offers Search Engine locally.

# Live Demo

A live demo of the Fetch Offers Search Engine is available at the following link:

[Live Demo Application](https://intelligentsearch.streamlit.app/)

Feel free to explore the live demo to see the search engine in action.


# Intelligent Fetch Offers Search Engine Solution Approach

Here we provide an overview of the project's approach, assumptions, and trade-offs. Additionally, a simplified flow diagram is provided for better understanding.

## Approach

The Fetch Offers Search Engine is designed to help users find relevant offers based on their search queries. Here's how it works:

1. **User Query Handling:**
   - Users can search for offers by either retailer, brand, or category.
   - Users are expected to provide the exact name of the retailer, brand, or category in their search query.
   - Basic typos in the user query are handled using edit distance/semantic search to ensure accurate matching.

2. **Exact Match - Retailer and Brand:**
   - The search engine attempts to find an almost exact match between the user query and the retailer or brand names.
   - The edit distance score is used as the relevancy score for these matches.

3. **Semantic Search - Category:**
   - If no relevant results are found in the previous step, the search engine performs a semantic search on the product category data.
   - The top k(for now set to 5) relevant categories are retrieved based on semantic similarity scores.
   - Brands associated with these categories are collected, as offers are available at the brand and retailer levels.

4. **Offer Retrieval - Category Level:**
   - Offers associated with the brands from the previous step are retrieved from the brand-to-offer mapping table.
   - To ensure a manageable list, the top 40 offers are considered.

5. **Re-Ranking for Relevance:**
   - The collected offers are re-ranked based on their relevance to the user's input query.
   - Relevance scores for brand/retailer matches are based on the edit distance score.
   - Relevance scores for category-level matches are a combination of semantic similarity scores and normalized re-ranking scores.
   - The final relevancy score for category-level offers is the reranked score weighted with semantic similarity score.

## Assumptions

To optimize system performance and user experience, the following assumptions were made:

1. **Single Query Type:** For relevant results users are expected to search for offers for either a brand, retailer, or category individually, not in combination. For example, they should search with keywords like "Target" or "Huggies," not sentences like "Huggies from Target."

2. **Search Query Content:** Users are expected to provide the name of a brand, retailer, or category in their search query. The system does not handle other types of text inputs. For example, they should search with keywords like "Target" not sentences like "get me offers from Target."

## Trade-Offs

Balancing accuracy and efficiency led to the following trade-offs:

- **Limited Context:** The search engine does not consider additional context within the query, such as user preferences or location. Advanced personalization features could be added but would require more processing and data.

- **Query Preprocessing:** The system relies on simple keyword level exact/semantic matching for efficiency. Handling more complex queries could be explored in future enhancements.

- **Semantic Search Resources:**
    The utilization of pre-trained deep encoders introduces a computational complexity to our search engine. The balance between model accuracy and response time is a crucial consideration, often contingent on the available computing resources.
    For extensive datasets or situations where real-time response is a priority, we offer an alternative approach. This involves the creation of an in-memory vector store, akin to popular databases like ChromaDB, Elasticsearch, and FAISS. These specialized vector databases efficiently store embeddings and enable us to perform searches based on nearest neighbors, as opposed to traditional substring searches. This can significantly expedite retrieval times.
    To demonstrate the use of vector stores and their integration with Langchain for information retrieval, I've included a Python script in the `src` directory: `retrieval_with_langchain.py`. This script showcases how vector stores can be harnessed to enhance the search and retrieval process, particularly when dealing with large-scale datasets.
    By implementing vector stores, we aim to provide an additional layer of optimization for users working with substantial datasets or those seeking to prioritize response time in their search experience.


## Flow Diagram

Fetch Offers Search Engine Flow
```
  +----------------+
  |    [Start]     |
  +-------|--------+
          |
          v
  +----------------+
  |  [User Query]  |
  +-------|--------+
          |
          v
  +----------------+
  | [Exact Match?] |
  |  /         \   |
  | Yes         No |
  |  \         /   |
  +----------------+
          |
          v
  +-----------------------------+
  | [Exact Match - Retailer/Brand] |
  +-----------------------------+
          |
          v
  +----------------------------+
  | [Semantic Search - Category]|
  +----------------------------+
          |
          v
  +----------------------------+
  | [Offer Retrieval - Brand Level]|
  +----------------------------+
          |
          v
  +------------------------+
  | [Re-Ranking for Relevance]|
  +------------------------+
          |
          v
  +-------------+
  |   [End]     |
  +-------------+

```

## Conclusion

The Fetch Offers Search Engine combines exact keyword matching with advanced semantic search techniques to provide users with relevant offers. Understanding the approach, assumptions, and trade-offs will help users make the most of the system. Further optimizations can enhance the user experience and accommodate more complex search queries.

Happy searching!
