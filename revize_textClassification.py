import spacy
import pandas as pd
import random
from collections import Counter
from spacy.util import minibatch
from spacy.training.example import Example

#load the data
data = pd.read_csv("/data/spam.csv")

#define the empty model
nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")
labels = Counter(data["label"].values).most_common()
for label in labels:
    textcat.add_label(label[0])

#training data preparation
train_text = data["text"].values
train_label = [{"cats": {"ham": label == "ham", "spam": label == "spam"}} for label in data["label"].values]
train_data = list(zip(train_text, train_label))

#Train the model
losses = {}
optimizer = nlp.begin_training()
random.seed(1)
random.shuffle(train_data)
for epoch in range(10):
    for batch in minibatch(train_data, size=8):
        for text, label in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, label)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(losses)

#try to predict
texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA" ]
docs = [nlp(text) for text in texts]
textcat = nlp.get_pipe("textcat")
scores = textcat.predict(docs)
predicted_label = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_label])



