import spacy
import pandas as pd
import random
from spacy.util import minibatch
from spacy.training.example import Example

#load data function
def load_data(csv_file, split=9):
    data = pd.read_csv(csv_file)
    train_data = data.sample(frac=1, random_state=7)
    split = int(len(train_data)*split)
    train_text = train_data.text.values[:split]
    test_text = train_data.text.values[split:]
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in train_data.sentiment.values]
    train_label = [{"cats": label} for label in labels[:split]]
    test_label = [{"cats": label} for label in labels[split:]]
    return train_text, train_label, test_text, test_label

#train model function
def train(model, train_data, optimizer, batch_size=8):
    losses = {}
    random.seed(1)
    random.shuffle(train_data)
    for batch in minibatch(train_data, batch_size):
        for text, label in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, label)
            model.update([example], sgd=optimizer, losses=losses)
    return losses

#predict function
def predict(nlp, texts):
    textcat = nlp.get_pipe("textcat")
    doc = [nlp(text) for text in texts]
    scores = textcat.predict(doc)
    predicted_label = scores.argmax(axis=1)
    return predicted_label

#evaluate function
def evaluate(nlp, texts, labels):
    prediction = predict(nlp, texts)
    true_label = [int(label["cats"]["POSITIVE"]) for label in labels]
    correct_prediction = prediction == true_label
    accuracy = correct_prediction.mean()
    return accuracy

nlp = spacy.blank("en")
#load data
csv_file = "/data/yelp_ratings.csv"
train_text, train_label, test_text, test_label = load_data(csv_file)
train_data = list(zip(train_text, train_label))

#define the empty model
textcat = nlp.add_pipe("textcat")
textcat.add_label("NEGATIVE")
textcat.add_label("POSITIVE")

#train the model
optimizer = nlp.begin_training()
losses = train(textcat, train_data, optimizer)
print(f"Loss: {losses:3f}")

#prediction
texts = test_text[34:38]
labels = test_label[34:38]
predicted_label = predict(nlp, texts)
for p, t in zip(predicted_label, texts):
    print(f"{textcat.labels[p]}: {t}\n")

#evaluate the model
accuracy = evaluate(nlp, texts, labels)
print(f"Accuracy: {accuracy:.3f}")
