import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_lg")
review_data = pd.read_csv("/data/yelp_ratings.csv")

reviews = review_data[:100]
with nlp.disable_pipes():
    vectors = np.array([nlp(review.text).vector for id, review in reviews.iterrows()])

vectors = np.load('/data/review_vectors.npy')

#split the dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(vectors, review_data, test_size=0.1, random_state=1)

#define the model and train it using svm
model = LinearSVC(random_state=1, dual=False)
model.fit(X_train, y_train)

#get the accuracy of the model svm
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy*100:.3f}%")

#define the classifier using knn with k=3
kmodel = KNeighborsClassifier(n_neighbors=3)
kmodel.fit(X_train, y_train)

#get the accuracy of the model knn
kaccuracy = kmodel.score(X_test, y_test)*100
print(f"Model accuracy for knn algorithm: {kaccuracy:.3f}%")



