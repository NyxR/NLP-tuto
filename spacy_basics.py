from pandas.io import html
import spacy
from spacy import displacy
from spacy.tokens import Span
#import textacy

nlp = spacy.load('en_core_web_sm')

text = """I absolutely love this place. The 360 degree glass windows with the 
            Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
            transports you to what feels like a different zen zone within the city. I know 
            the price is slightly more compared to the normal American size, however the food 
            is very wholesome, the tea selection is incredible and I know service can be hit 
            or miss often but it was on point during our most recent visit. Definitely recommend!

            I would especially recommend the butternut squash gyoza."""

doc = nlp(text)

#get all the sentenses in the doc
sentences = list(doc.sents)

#get all the name entities in the doc or a sentense (NER)
ents = list(sentences[0].ents)
ents = list(doc.ents)
for ent in ents:
    print(ent.text, ent.label_)

#get the POS (Part of Speech)
sentence = sentences[0]
for token in sentence:
    print(token.text, token.pos_)

#extract nouns and chunks
nouns = []
for token in sentence:
    if token.pos_ == "NOUN":
        nouns.append(token)

#get the doc noun chunks
chunks = list(doc.noun_chunks)
for chunk in chunks:
    if "watch" == str(chunk):
        print(chunk)

#get the verb using matches patterns
#patterns = [{"POS": "VERB"}]
#verb_phrases = textacy.extract.matches(doc, patterns=patterns)
#for verb_phrase in verb_phrases:
#    print(verb_phrase)

#get the verb
verbs = []
for token in sentence:
    if token.pos_ == "VERB":
        verbs.append(token)
print(verbs)

#get the lemma or the root of the verb in sentence
for word in sentence:
    if word.pos_ == "VERB":
        print(word, word.lemma_)

#data visualization using displacy return html file
html = displacy.render(sentence, style='dep') #style=dep or style=ent
with open("data_vis.html", 'w') as f:
    f.write(html)

#add new entitie
doc1 = nlp("fb is hiring a new vice president of global policy")
ents = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc1.ents]
print("Before:", ents)

new_ent = Span(doc1, 0, 1, label="ORG")
orig_ents = doc1.ents
doc1.set_ents([new_ent], default="unmodified")
#Or
doc1.ents = orig_ents + [new_ent]
ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc1.ents]
print("After: ", ents)



