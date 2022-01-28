
import spacy

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("merge_entities")
nlp.add_pipe("merge_noun_chunks")

TEXTS = [
    "Net income was $9.4 million compared to the prior year of $2.7 million.",
    "Revenue exceeded twelve billion dollars, with a loss of $1b.",
]

#output Net income --> $9.4 

for doc in nlp.pipe(TEXTS):
    for token in doc:
        if token.ent_type == "MONEY":
            if token.dep_ in ("attr", "dobj"):
                subj = [w for w in token.head.lefts if w.dep_ == "nsubj"]
                if subj:
                    print(subj, " --> ", token)
            elif token.dep_ == "pobj" and token.head.dep_ == "prep":
                print(token.head.head, " --> ", token)
