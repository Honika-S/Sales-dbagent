
import pandas as pd
import streamlit as st
from pymongo import MongoClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from bson.decimal128 import Decimal128
import pymongo
from bson import SON

# MongoDB setup
uri = "mongodb+srv://honikasankar:honi@cluster0.p2s1i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)

# Hugging Face embeddings setup
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ollama = Ollama(model="llama3.2", base_url="http://35.209.140.5/")

# Sample approved email IDs
approved_emails = ["honikasankar@gmail.com", "user2@example.com"]

# Helper function to fetch data from MongoDB and convert Decimal128 to float
def fetch_data(collection):
    result = collection.find().limit(100)
    df = pd.DataFrame(list(result))
    # Convert Decimal128 columns to float
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: float(x.to_decimal()) if isinstance(x, Decimal128) else x)
    return df

# Embed a query and find the most relevant documents across all fields
def find_relevant_docs(query, df):
    # Embed the query
    query_embedding = embedding_model.encode([query])
    
    # Concatenate all string fields into a single text for embedding
    df['combined_text'] = df.astype(str).agg(' '.join, axis=1)
    
    # Embed each text entry in the dataset
    text_embeddings = embedding_model.encode(df['combined_text'].fillna('').tolist())
    
    # Calculate cosine similarity and get top match
    similarities = cosine_similarity(query_embedding, text_embeddings).flatten()
    top_match_index = np.argmax(similarities)
    return df.iloc[top_match_index]

# Streamlit app
st.title("Hybrid Database Q&A and Visualization System")

# List available databases
st.subheader("Available Databases")
databases = client.list_database_names()
selected_db = st.selectbox("Select a Database:", databases)

# List collections in the selected database
if selected_db:
    db = client[selected_db]
    collections = db.list_collection_names()
    selected_collection_name = st.selectbox("Select a Collection:", collections)

    # Define the selected collection based on user input
    if selected_collection_name:
        collection = db[selected_collection_name]

        # Query Section
        query = st.text_input("Ask a question about the data:")
        if st.button("Submit Query") and query:
            df = fetch_data(collection)
            
            # Find relevant doc
            relevant_doc = find_relevant_docs(query, df)
            
            # Use Ollama to generate a detailed answer
            prompt = f"Given the following data, answer the question:\n\nData: {relevant_doc.to_dict()}\n\nQuestion: {query}\n\nAnswer:"
            answer = ollama(prompt)
            
            # Display the answer
            st.write("Answer:", answer)

# User access section
st.subheader("Access Verification")
email = st.text_input("Enter your email to access visualization features:")

# Check access and store session state
if st.button("Verify Access"):
    if email in approved_emails:
        st.session_state.access_granted = True
        st.success("Access granted! You can now modify visualizations.")
    else:
        st.session_state.access_granted = False
        st.error("Access denied! You are not authorized.")

# Visualization Section
if "access_granted" in st.session_state and st.session_state.access_granted and selected_collection_name:
    st.subheader("Data Visualization Options")

    # Fetch data for visualization
    df = fetch_data(collection)
    
    # Filter columns for numeric data
    numeric_fields = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_fields:
        st.warning("No numeric data available for visualization.")
    else:
        # Select a numeric field to visualize
        field = st.selectbox("Select Field to Visualize:", numeric_fields)
        plot_type = st.selectbox("Select Plot Type:", ['Histogram', 'Boxplot'])
        bins = st.slider("Number of Bins (for Histogram):", min_value=5, max_value=50, value=10)

        # Generate the visualization
        if st.button("Generate Visualization"):
            plt.figure(figsize=(10, 6))

            if plot_type == 'Histogram':
                plt.hist(df[field].dropna(), bins=bins, color='skyblue', edgecolor='black')
                plt.xlabel(field.capitalize())
                plt.ylabel("Frequency")
                plt.title(f"{field.capitalize()} Distribution")

            elif plot_type == 'Boxplot':
                plt.boxplot(df[field].dropna(), vert=False, patch_artist=True)
                plt.xlabel(field.capitalize())
                plt.title(f"{field.capitalize()} Boxplot")

            st.pyplot(plt)
else:
    st.warning("Please verify your access to use visualization features.")
