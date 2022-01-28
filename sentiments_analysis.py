import pandas as pd
import spacy
import random
from spacy.util import minibatch
from spacy.training.example import Example


#define function load_data for data spliting
def load_data(csv_file, split=0.9):
    data = pd.read_csv(csv_file)
    #shuffle the data
    train_data = data.sample(frac=1, random_state=7)

    texts = train_data.text.values
    labels = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} 
                for y in train_data.sentiment.values]
    split = int(len(train_data) * split)

    train_label = [{'cats': labels} for labels in labels[:split]]
    val_label = [{'cats': labels} for labels in labels[split:]]

    return texts[:split], train_label, texts[split:], val_label

#define train function
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

#making prediction
def predict(nlp, texts):
    docs = [nlp.tokenizer(text) for text in texts]
    textcat = nlp.get_pipe("textcat")
    scores = textcat.predict(docs)
    predict_label = scores.argmax(axis=1)
    return predict_label

#define evaluate function
def evaluate(model, texts, labels):
    predicted_class = predict(model, texts) #type numpy array!!!
    #convert the label like (POSITIVE -> 1, NEGATIVE -> 0)
    true_class = [int(each['cats']['POSITIVE']) for each in labels]
    #correct prediction
    correct_prediction = predicted_class == true_class
    #get the accuracy
    accuracy = correct_prediction.mean() #type numpy array
    return correct_prediction, accuracy

#load the data
train_text, train_label, val_text, val_label = load_data("yelp_ratings.csv")
train_data = list(zip(train_text, train_label))

#define the model
nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")

textcat.add_label("NEGATIVE")
textcat.add_label("POSITIVE")

spacy.util.fix_random_seed(1)
random.seed(1)

#define the optimizer
optimizer = nlp.begin_training()

#train the model and get the losses
losses = train(nlp, train_data, optimizer)
print(losses['textcat'])

#define the validation text to use
texts = val_text[34:38]
labels = val_label[34:38]
predictions = predict(nlp, texts)

for p, t in zip(predictions, texts):
    print(f"{textcat.labels[p]}: {t} \n")

#Iteration
n_iters = 5
for i in range(n_iters):
    losses = train(nlp, train_data, optimizer)
    correct_prediction, accuracy = evaluate(nlp, texts, labels)
    print(f"Loss: {losses['textcat']:.3f}, Accuracy: {accuracy:.3f}")

