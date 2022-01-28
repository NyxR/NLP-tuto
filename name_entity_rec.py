import spacy
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")
doc = nlp("fb is hiring a new vice president of global policy")
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc.ents]
print("Before: ", ents)
new_ent = Span(doc, 0, 1, label="ORG")
orig_ents = list(doc.ents)
doc.ents = orig_ents + [new_ent]
ents = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
print("After: ", ents)