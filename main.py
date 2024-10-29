
#all working correctly code
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from bson.decimal128 import Decimal128
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.prompts import PromptTemplate
import re
from langchain_community.llms import Ollama
import nltk  # Make sure to install nltk if not already installed
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (if not done already)
nltk.download('punkt')
nltk.download('stopwords')

# Replace this with your MongoDB Atlas URI
uri = "mongodb+srv://honikasankar:honi@cluster0.p2s1i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri)
db = client['sample_airbnb']
collection = db['listingsAndReviews']

# Sample approved email IDs
approved_emails = ["honikasankar@gmail.com", "user2@example.com"]

# Function to fetch data from MongoDB
def fetch_data():
    result = collection.find().limit(100)
    return pd.DataFrame(list(result))

# Function to convert Decimal128 to float in DataFrame
def convert_decimal_columns(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: float(x.to_decimal()) if isinstance(x, Decimal128) else x)
    return df

# Function to visualize data
def visualize_data(df, plot_type='histogram', field='price', bins=50):
    df = convert_decimal_columns(df)
    plt.figure(figsize=(10, 6))
    
    if plot_type == 'histogram':
        df[field].hist(bins=bins)
        plt.title(f'{field.capitalize()} Distribution')
        plt.xlabel(field.capitalize())
        plt.ylabel('Frequency')
        
    elif plot_type == 'boxplot':
        df.boxplot(column=field)
        plt.title(f'{field.capitalize()} Boxplot')
        plt.ylabel(field.capitalize())
        
    st.pyplot(plt)

# Langchain & Ollama Integration
ollama = Ollama(model="llama3.2", base_url="http://35.209.140.5/")

# Prompt template
prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are an agent for MongoDB queries. The user asks: {query}.
    Fetch relevant data from MongoDB and return the results.
    """
)

# Function to process the full sentence query
def process_query(query):
    # Tokenize the query and remove stopwords
    tokens = word_tokenize(query.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return filtered_tokens

# Query MongoDB
def query_mongodb(query: str):
    try:
        keywords = process_query(query)
        if not keywords:
            return pd.DataFrame()  # Return empty DataFrame if no keywords found
        
        # Construct regex queries for each keyword
        regex_conditions = [{"$or": [
            {"address.market": {"$regex": keyword, "$options": "i"}},
            {"name": {"$regex": keyword, "$options": "i"}},
            {"description": {"$regex": keyword, "$options": "i"}}
        ]} for keyword in keywords]
        
        # Combine the conditions into a single query
        final_query = {"$or": regex_conditions}
        
        results = collection.find(final_query).limit(5)  # Increase limit as needed
        return pd.DataFrame(list(results))
    except Exception as e:
        st.error(f"Error querying the database: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Define toolset
tools = [
    Tool(
        name="query_mongodb",
        func=query_mongodb,
        description="Query the MongoDB database"
    )
]

# Initialize agent
agent = initialize_agent(tools, ollama, agent_type="zero-shot-react-description", verbose=True)

# Streamlit UI components
st.title("Airbnb Data Query & Visualization System")

# Query Section
st.subheader("Natural Language Query to MongoDB")
query = st.text_input("Ask a question about the Airbnb data:")

if st.button("Submit Query"):
    if query:
        response_df = query_mongodb(query)
        if not response_df.empty:
            st.write("Query Result:")
            st.write(response_df)
        else:
            st.write("No results found for the given query.")
    else:
        st.warning("Please enter a query.")

# Access Authorization Section with Session State
st.subheader("Access Data Visualization")
if "access_granted" not in st.session_state:
    st.session_state.access_granted = False

email = st.text_input("Enter your email ID for access verification:")

if st.button("Verify Access"):
    if email:
        is_authorized = email in approved_emails
        if is_authorized:
            st.session_state.access_granted = True
            st.success("You are authorized for data visualization.")
        else:
            st.session_state.access_granted = False
            st.error("Unauthorized user. You do not have access to the data visualization feature.")
    else:
        st.warning("Please enter an email ID.")

# Display Visualization Options if Access is Granted
if st.session_state.access_granted:
    st.subheader("Data Visualization Options")
    plot_type = st.selectbox("Select Plot Type:", ['histogram', 'boxplot'])
    field = st.selectbox("Select Field to Visualize:", ['price', 'accommodates', 'bedrooms', 'bathrooms'])
    bins = st.slider("Select Number of Bins:", min_value=1, max_value=100, value=50)

    # Generate visualization when button is clicked
    if st.button("Generate Visualization"):
        df = fetch_data()
        visualize_data(df, plot_type, field, bins)
