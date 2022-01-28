import spacy
from spacy.symbols import nsubj, VERB

nlp = spacy.load('en_core_web_sm')
doc = nlp("Autonomous cars shift insurance liability toward manufacturers")


#Finding possible verb with a subject below

verb = set()
for possible_subject in doc:
    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
        verb.add(possible_subject.head)
print(verb)