import numpy as np
import pandas as pd
import spacy

def cosine_similarity(a, b):
    return np.dot(a, b)/np.sqrt(a.dot(a)*b.dot(b))

nlp = spacy.load("en_core_web_lg")

review_data = pd.read_csv('/data/yelp_ratings.csv')
vectors = np.load('/data/review_vectors.npy')

review = """I absolutely love this place. The 360 degree glass windows with the 
            Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
            transports you to what feels like a different zen zone within the city. I know 
            the price is slightly more compared to the normal American size, however the food 
            is very wholesome, the tea selection is incredible and I know service can be hit 
            or miss often but it was on point during our most recent visit. Definitely recommend!

            I would especially recommend the butternut squash gyoza."""

review_vec = nlp(review).vector
vec_mean = vectors.mean(axis=0)
#get the centered vector
centered = review_vec - vec_mean

#get all the similarity score within the dataset review_data
sims = [cosine_similarity(vector - vec_mean, centered) for vector in vectors]

#get the most similar document index from the dataset
most_similar = sims.index(max(sims))

#show the most similar document text
print(review_data.iloc[most_similar].text)