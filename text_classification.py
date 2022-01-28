import pandas as pd
import spacy
import random
from collections import Counter
from spacy.util import minibatch
from spacy.training.example import Example

#loading spam data
#ham mean non-spam messages
spam = pd.read_csv("data/spam.csv")
#print(spam.head(10))

#create an empty model
nlp = spacy.blank("en")

#add the TextCategorizer to the empty model
textcat = nlp.add_pipe("textcat")

#adding labels to text classifier
labels_list = Counter(spam["label"]).most_common()
for label in labels_list:
    textcat.add_label(label[0])

#define the train set (train_label and train_text)
train_texts = spam["text"].values
train_labels = [{'cats': {'ham': label == 'ham', 
                        'spam': label == 'spam'}}
                for label in spam['label']]

train_data = list(zip(train_texts, train_labels))

#train the model
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training() #setting the optimizer

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    #Create the batch generator with size = 8
    batches = minibatch(train_data, size=8)
    #Iterate through batches
    for batch in batches:
        for text, label in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, label)
            nlp.update([example], sgd=optimizer, losses=losses)
    print(losses)

#try to predict

texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA" ]
docs = [nlp.tokenizer(text) for text in texts]

textcat = nlp.get_pipe('textcat')
scores = textcat.predict(docs)
print(scores)

#From the scores, find the highest score/propability
predicted_label = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_label])

