# DB agent

## Overview

This application is a hybrid database question-and-answer system combined with data visualization capabilities. It allows users to query data from a MongoDB database and receive detailed answers generated by the LLaMA model from Ollama. Additionally, users can visualize different aspects of the dataset through various plot types.

## Technologies Used

- **Streamlit**: For creating the web application.
- **Pandas**: For data manipulation and analysis.
- **MongoDB**: As the database for storing and retrieving Airbnb listings data.
- **LangChain**: For handling embeddings and LLaMA model interactions.
- **Sentence Transformers**: To generate embeddings for text queries and descriptions.
- **Matplotlib**: For creating visualizations of the data.

## Set Up MongoDB

To connect the application to MongoDB, follow these steps:

1. **Create a MongoDB Account**:
   - Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and sign up for a free account.

2. **Create a Cluster**:
   - Once logged in, create a new cluster by following the prompts (choose the free tier for testing).

3. **Create a Database**:
   - After your cluster is set up, navigate to the **Database** section and create a new database named `sample_airbnb`.

4. **Add Collections**:
   - Within the `sample_airbnb` database, create a collection named `listingsAndReviews`.

5. **Add Sample Data**:
   - Insert sample data into the `listingsAndReviews` collection. You can find sample Airbnb data online or create your own documents.

6. **Get Connection URI**:
   - Click on the **Connect** button in your cluster dashboard.
   - Choose **Connect your application**.
   - Copy the connection string (URI) provided and replace `<username>` and `<password>` with your MongoDB Atlas username and password. Ensure the `retryWrites=true&w=majority` part is included.
