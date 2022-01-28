from os import set_inheritable
import pandas as pd
import numpy as np
import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example

#define the load data function
def load_data(csv_file, split=0.9):
    data = pd.read_csv(csv_file)
    train_data = data.sample(frac=1, random_state=7)
    split = int(len(train_data)*split)
    train_text = train_data.text.values[:split]
    test_text = train_data.text.values[split:]
    labels = [{'POSITIVE': bool(y), 'NEGATIVE': not bool(y)}
              for y in train_data.sentiment.values]
    train_label = [{'cats': label} for label in labels[:split]]
    test_label = [{'cats': label} for label in labels[split:]]

    return train_text, train_label, test_text, test_label

#define the train function
def train(model, train_data, optimizer, batch_size=8):
    losses = {}
    random.seed(1)
    random.shuffle(train_data)
    for batch in minibatch(train_data, size=batch_size):
        for text, label in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, label)
            model.update([example], sgd=optimizer, losses=losses)
    return losses

#define the predict function
def predict(nlp, texts):
    docs = [nlp(text) for text in texts]
    textcat = nlp.get_pipe('textcat')
    scores = textcat.predict(docs)
    predicted_class = scores.argmax(axis=1)
    return predicted_class

#define the evaluate function
def evaluate(nlp, texts, labels):
    prediction = predict(nlp, texts)
    true_class = [int(label['cats']['POSITIVE']) for label in labels]
    correct_prediction = prediction == true_class
    accuracy = correct_prediction.mean()
    return correct_prediction, accuracy


#load data
csv_file = '/data/yelp_ratings.csv'
train_text, train_label, test_text, test_label = load_data(csv_file)
train_data = list(zip(train_text, train_label))

#prepare the model
nlp = spacy.blank('en')
textcat = nlp.add_pipe('textcat')
textcat.add_label('NEGATIVE')
textcat.add_label('POSITIVE')

#train the model
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()
losses = train(textcat, train_data, optimizer)

#test the model
texts = test_text[34:38]
labels = test_label[34:38]
predictions = predict(nlp, texts)
for p, t in zip(predictions, texts):
    print(f"{textcat.labels[p]} : {t}\n")

#evaluate the model
correct_prediction, accuracy = evaluate(nlp, texts, labels)
print(f"Correct prediction {correct_prediction} : accuracy: {accuracy:.3f}")

#train and evaluate the model with iteration = 5
n_iters = 5
for i in n_iters:
    losses = train(textcat, train_data, optimizer)
    correct_prediction, accuracy = evaluate(nlp, texts, labels)
    print(f"Loss: {losses['textcat']:.3f}, Accuracy: {accuracy:.3f}")