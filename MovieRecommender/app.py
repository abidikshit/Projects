import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from fastapi import FastAPI, HTTPException

# Initialize FastAPI app
app = FastAPI()

# Fetch data from movielens
data = fetch_movielens(min_rating=3.0)

# Create the recommendation model
model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)

# Recommendation function
def recommendation(user_id):
    n_users, n_items = data['train'].shape

    if user_id < 0 or user_id >= n_users:
        raise HTTPException(status_code=400, detail="Invalid user ID")

    known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

    scores = model.predict(user_id, np.arange(n_items))

    top_items = data['item_labels'][np.argsort(-scores)]

    return {
        "user_id": user_id,
        "known_positives": known_positives[:3].tolist(),
        "recommended": top_items[:3].tolist()
    }

# Define an endpoint to receive user input and return recommendations
@app.post("/recommend/{user_id}")
async def get_recommendations(user_id: int):
    recommendations = recommendation(user_id)
    return recommendations
