import streamlit as st
from pymongo import MongoClient
from langchain_ollama import ChatOllama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import ast
from bson.decimal128 import Decimal128

# Initialize MongoDB connection with updated URI
MONGODB_URI = "mongodb+srv://honikasankar:honi@cluster0.p2s1i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGODB_URI)

# Initialize Llama instance
ollama = ChatOllama(model="llama3.2", base_url="http://35.209.140.5/")

# Set up a prompt template for Llama to generate MongoDB queries
prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template="You are a MongoDB query assistant. Interpret the user's query: '{user_query}'. "
              "Identify the database and collection, and generate a MongoDB filter to retrieve relevant data. "
              "Respond with only the database name, collection name, and filter as a dictionary in Python format."
)

# Set up the LLM chain
llm_chain = LLMChain(llm=ollama, prompt=prompt_template)
# Function to execute MongoDB queries
def execute_query(database_name, collection_name, query):
    try:
        db = client[database_name]
        collection = db[collection_name]

        # Original filter handling restored
        filter_query = query.get("filter", {})

        # Handle edge cases for filter query ('{', None, {})
        if filter_query == '{' or filter_query is None or not isinstance(filter_query, dict):
            filter_query = {}

        sort_query = query.get("sort", None)
        limit_value = query.get("limit", None)

        cursor = collection.find(filter_query)

        if sort_query:
            sort_fields = [(field, direction) for field, direction in sort_query.items()]
            cursor = cursor.sort(sort_fields)

        if limit_value:
            cursor = cursor.limit(limit_value)

        result = list(cursor)
        return result if result else "No data found."
    except Exception as e:
        return str(e)

def flatten_result(result):
    """Flatten result if it has nested structures to make it DataFrame compatible."""
    flattened_result = []
    for item in result:
        flattened_item = {}
        for key, value in item.items():
            if isinstance(value, list) or isinstance(value, dict):
                flattened_item[key] = str(value)  
            else:
                flattened_item[key] = value
        flattened_result.append(flattened_item)
    return flattened_result

def generate_report(df):
    report = []
    
    for col in df.columns:
        # Handle Decimal128 conversion to float
        if df[col].apply(lambda x: isinstance(x, Decimal128)).any():
            try:
                df[col] = df[col].apply(lambda x: float(x.to_decimal()) if isinstance(x, Decimal128) else x)
            except Exception:
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, Decimal128) else x)

        if df[col].apply(lambda x: isinstance(x, list) or isinstance(x, dict)).any():
            report.append(f"The column '{col}' contains lists or dictionaries, so unique value calculations are skipped.")
            continue  
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            most_common = df[col].mode()[0]
            report.append(f"The column '{col}' has {unique_count} unique values with the most common being '{most_common}'.")
        else:
            report.append(f"The column '{col}' has a mean value of {df[col].mean():.2f}, a minimum value of {df[col].min()}, and a maximum value of {df[col].max()}.")
    
    return " ".join(report)

st.title("MongoDB Query Chatbot")

user_query = st.text_input("Ask your question about the database (specifying both the database and collection):")
run_query = st.button("Submit")
def handle_user_query(user_query):
    if not user_query:
        return "Please ask a question."
    mongo_query_response = llm_chain({"user_query": user_query})
    mongo_query_text = mongo_query_response.get("text", "").strip() 

    if not mongo_query_text:
        return "The model did not return a valid query."

    # Attempt to parse the generated query (expecting a dictionary format)
    try:
        # Safely evaluate response as a dictionary
        mongo_query_dict = ast.literal_eval(mongo_query_text)

        # Extract database, collection, and filter information
        database_name = mongo_query_dict.get("database")
        collection_name = mongo_query_dict.get("collection")
        query = mongo_query_dict.get("filter", {})

        # Validate that database and collection names are provided
        if not database_name or not collection_name:
            return "The model did not specify a database or collection name."

        # Connect to the specified database and collection
        db = client[database_name]
        collection = db[collection_name]
        sample_document = collection.find_one()
        #for viewing fields
        # if sample_document:
        #     field_names = list(sample_document.keys())
        #     st.write("Fields recognized in the collection:", field_names)

        # Check if the query is asking for unique values in a specific field
        if "unique" in user_query.lower():
            # Extract the target field name from the query
            field_name = None
            for field in collection.find_one().keys():
                if field in user_query:
                    field_name = field
                    break

            if field_name:
                # Retrieve unique values for the specified field
                unique_values = collection.distinct(field_name)
                return f"Unique values in '{field_name}' field: {unique_values}"
            else:
                return "Could not identify the specific field for unique values. Please check the query."
        # Execute the query and get results
        results = execute_query(database_name, collection_name, query)

        # If results are found, display them
        if results:
            # Convert results to DataFrame for easier presentation
            df = pd.DataFrame(flatten_result(results))

            # Generate a report if the query specifies it, otherwise show the DataFrame
            if "report" in user_query.lower() or "summary" in user_query.lower():
                return generate_report(df)
            else:
                return df
        else:
            return "No data found for the specified query."

    except (ValueError, SyntaxError) as e:
        return f"There was an error processing the MongoDB query: {e}. Response was: {mongo_query_text}"

if run_query:
    response = handle_user_query(user_query)
    
    # Check if the response is a DataFrame (i.e., valid MongoDB result)
    if isinstance(response, pd.DataFrame):
        st.dataframe(response)  # Display the result as a table in Streamlit
    else:
        st.text_area("Report", response, height=300)  # Display the result as a text report

