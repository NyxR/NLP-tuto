import numpy as np
import pandas as pd
import spacy
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

nlp = spacy.load('en_core_web_sm')

#load data
review_data = pd.read_csv('/data/yelp_ratings.csv')
vectors = np.load('/data/review_vectors.npy')

#split data
X_train, y_train, X_test, y_test = train_test_split(vectors, review_data, test_size=0.1, random_state=1)

#train the model
model = LinearSVC(random_state=1, dual=False)
model.fit(X_train, y_train)

#get the accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy*100:.2f}")

#Document similarity
#define the cosine_similarity function
def cosine_similarity(a, b):
    return np.dot(a,b)/np.sqrt(a.dot(a)*b.dot(b))

review = """I absolutely love this place. The 360 degree glass windows with the 
            Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
            transports you to what feels like a different zen zone within the city. I know 
            the price is slightly more compared to the normal American size, however the food 
            is very wholesome, the tea selection is incredible and I know service can be hit 
            or miss often but it was on point during our most recent visit. Definitely recommend!

            I would especially recommend the butternut squash gyoza."""

#get the mean of vectors
vec_mean = vectors.mean()
review_vec = nlp(review).vector

#get the centered of review vector
centered = review_vec - vec_mean

#get the similarity score of this review for each document in the dataset
sims = [cosine_similarity(vector - vec_mean, centered) for vector in vectors]

#get the most similar document
most_similar = sims.index(max(sims))
print(f"{review_data.iloc[most_similar].text}")