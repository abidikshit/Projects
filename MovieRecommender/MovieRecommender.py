import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data from model
data = fetch_movielens(min_rating = 3.0)

#print training and testing data
print(repr(data['train']))
print(repr(data['test']))

#create model
model = LightFM(loss = 'warp')

#train mode
model.fit(data['train'], epochs=30, num_threads=2)

#recommender fucntion
def recommendation(model, data, user_id):
    n_users, n_items = data['train'].shape

    known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

    scores = model.predict(user_id, np.arange(n_items))

    top_items = data['item_labels'][np.argsort(-scores)]
    
    print("User %s" % user_id)
    print("     Known positives:")
    
    for x in known_positives[:3]:
        print("        %s" % x)
    
    print("     Recommended:")
    
    for x in top_items[:3]:
        print("        %s" % x)
            
if __name__ == "__main__":
    user_id = int(input("Enter user ID: "))  # Input the user ID from the terminal
    # Replace 'model', 'data', and 'user_id' with your actual recommendation model, data, and user ID
    recommendation(model, data, user_id)