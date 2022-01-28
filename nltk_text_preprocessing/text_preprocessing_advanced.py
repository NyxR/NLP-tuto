from itertools import chain
import nltk
import string
import re
from nltk import chunk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

#Text Preprocessing step-2

#Part of Speech tagging
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
def pos_tagging(text):
    word_tokens = word_tokenize(text)
    return pos_tag(word_tokens)
pos_tagging('You just gave me a scare')

#Chunking
def chunking(text, grammar):
    word_tokens = word_tokenize(text)
    word_pos = pos_tag(word_tokens)
    chunkParser = nltk.RegexpParser(grammar)
    tree = chunkParser.parse(word_pos)
    for subtree in tree.subtrees():
        print(subtree)
    return tree.draw()

sentence = 'the little yellow bird is flying in the sky'
grammar = "NP: {<DT>?<JJ>*<NN>}" 
# This rule says that an NP (Noun Phrase) chunk should 
# be formed whenever the chunker finds an optional determiner (DT) 
# followed by any number of adjectives (JJ) and then a noun (NN).
chunking(sentence, grammar)

#Name Entity Recognition (NER)
def named_entity_recognition(text):
    word_tokens = word_tokenize(text)
    word_pos = pos_tag(word_tokens)
    print(ne_chunk(word_pos))

text = 'anadama'
named_entity_recognition(text)

