import spacy
from spacy.tokens import doc

nlp = spacy.load("en_core_web_md")

doc1 = nlp(nlp("good")[0].lemma_)
doc2 = nlp(nlp("best")[0].lemma_)

print(f"{doc1} <-> {doc2}, similarity:{doc1.similarity(doc2)}")
