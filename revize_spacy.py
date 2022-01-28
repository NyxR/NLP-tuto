import spacy

nlp = spacy.load("en_core_web_sm")
text = """I absolutely love this place. The 360 degree glass windows with the 
            Yerba buena garden view, tea pots all around and the smell of fresh tea everywhere 
            transports you to what feels like a different zen zone within the city. I know 
            the price is slightly more compared to the normal American size, however the food 
            is very wholesome, the tea selection is incredible and I know service can be hit 
            or miss often but it was on point during our most recent visit. Definitely recommend!

            I would especially recommend the butternut squash gyoza."""
doc = nlp(text)

#extract sentences
sentences = list(doc.sents)
#extract entities
entities = list(doc.ents)
for ent in entities:
    print(ent.text, ent.label_)

sentence = sentences[0]
#get Part of Speech
for token in sentence:
    print(token.text, token.pos_)

#get the noun
nouns = []
for token in sentence:
    if token.pos_ == "NOUN":
        nouns.append(token)

#get doc noun chunks
chunks = list(doc.noun_chunks)
for chunk in chunks:
    if "watch" == chunk:
        print(chunk)

#get the verb
verbs = []
for token in sentence:
    if token.pos_ == "VERB":
        verbs.append(token)

#get the lemma of all the verbs
lemmas = []
for word in sentence:
    if word.pos_ == "VERB":
        lemmas.append([word, word.lemma_])

        
