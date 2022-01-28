import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("abasdlfkjh")
for ent in doc.ents:
    print(ent.text, ent.label_)

