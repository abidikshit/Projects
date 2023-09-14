**Movie Recommendation System**

This code utilizes the lightfm recommender system library to build and train a hybrid algorithm that combines content-based and collaborative filtering approaches. It employs the WARP (Weighted Approximate-Rank Pairwise) loss function on the Movielens dataset, which comprises movies and user ratings from a diverse user base of over 1700 individuals. After the model is trained, the script generates and outputs movie recommendations for selected users from the dataset to the terminal.

## Prerequisites

Before running the script, ensure that you have the following dependencies installed:

- numpy (http://www.numpy.org/)
- scipy (https://www.scipy.org/)
- lightfm (https://github.com/lyst/lightfm)
- uvicorn (https://www.uvicorn.org)
- fastapi (https://fastapi.tiangolo.com)

You can install any missing dependencies using pip.

## Usage
To deploy your collaborative filtering recommender system using FastAPI, you'll need to create a FastAPI application that serves as the user interface for your recommendation model. Here are the steps to deploy it:

1. Install FastAPI ```pip install fastapi```
2. Create a FastAPI Application ```app.py```
3. Install Uvicorn ```pip install uvicorn```
4. Run the FastAPI App in the same directory ```uvicorn app:app --host 0.0.0.0 --port 8000 --reload```
5. Access the UI ```http://localhost:8000/recommend/{user_id}```

Once the application is running, you can make POST requests to http://localhost:8000/recommend/{user_id} to get recommendations for a specific user ID.

This will initiate the movie recommendation system, allowing you to receive movie recommendations based on the trained hybrid algorithm.