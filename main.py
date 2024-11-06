import pandas as pd
import streamlit as st
from pymongo import MongoClient
from langchain.llms import Ollama
from sentence_transformers import SentenceTransformer
from bson.decimal128 import Decimal128
import re

# MongoDB setup
uri = "mongodb+srv://honikasankar:honi@cluster0.p2s1i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)

# Model setup
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
ollama = Ollama(model="llama3.2", base_url="http://35.209.140.5/")

# Helper function to flatten documents and convert Decimal128 to float
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ', '.join(map(str, v))))
        elif isinstance(v, Decimal128):
            items.append((new_key, float(v.to_decimal())))
        else:
            items.append((new_key, v))
    return dict(items)

def fetch_data(collection, limit=100):
    result = collection.find().limit(limit)
    data = [flatten_dict(doc) for doc in result]
    return pd.DataFrame(data)

# Function to generate report in paragraph format
def generate_report(df):
    report = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            most_common = df[col].mode()[0]
            report.append(f"'{col}' has {unique_count} unique values, most common: '{most_common}'.")
        else:
            report.append(f"'{col}' has mean: {df[col].mean():.2f}, min: {df[col].min()}, max: {df[col].max()}.")
    return " ".join(report)

# NLP-based query parsing function with intent detection
def parse_query(query):
    db_name = col_name = None
    report_flag = summary_flag = records_flag = False
    
    db_match = re.search(r'(?:from\s+the\s+)?([\w-]+)\s+database|database\s+([\w-]+)', query, re.IGNORECASE)
    col_match = re.search(r'(?:and\s+)?([\w-]+)\s+collection|collection\s+([\w-]+)', query, re.IGNORECASE)

    if db_match: db_name = db_match.group(1)
    if col_match: col_name = col_match.group(1)
    
    # Determine intent based on keywords
    if "report" in query.lower():
        report_flag = True
    elif "summary" in query.lower() or "summarize" in query.lower():
        summary_flag = True
    elif "records" in query.lower() or "data" in query.lower():
        records_flag = True
    
    return db_name, col_name, report_flag, summary_flag, records_flag

# Streamlit app setup
st.title("Database Query Chatbot")

query = st.text_input("Ask a question about the data:")

if st.button("Submit Query") and query:
    db_name, collection_name, report_request, summary_request, records_request = parse_query(query)
    
    if db_name and collection_name:
        try:
            db = client[db_name]
            collection = db[collection_name]
            df = fetch_data(collection, limit=5)
            
            if report_request and not df.empty:
                report = generate_report(df)
                st.write(f"Ok, here is the report: {report}")
            
            elif summary_request and not df.empty:
                # Generate a summary using LLM
                data_summary = ollama(query)
                st.write(f"Ok, here is the summary: {data_summary}")
            
            elif records_request and not df.empty:
                st.write(f"Here are the first records from `{db_name}` -> `{collection_name}`:")
                st.write(df.head())
            
            else:
                st.write(f"No matching records found or unrecognized intent for `{db_name}` -> `{collection_name}`.")
                
        except Exception as e:
            st.error(f"Error accessing `{db_name}` -> `{collection_name}`: {e}")
    
    else:
        st.write("Please specify both the database and collection in your query.")


