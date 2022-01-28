import re
import string
import inflect
import nltk
from nltk import tree
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#text to lowercase
def text_lowercase(text):
    return text.lower()

#remove number
def remove_number(text):
    result = re.sub(r'\d+', '', text)
    return result

#remove whitespace
def remove_whitespace(text):
    return " ".join(text.split())

#number to text
p = inflect.engine()
def number_to_text(text):
    temps = text.split()
    new_str = []
    for temp in temps:
        if temp.isdigit():
            new_str.append(p.number_to_words(temp))
        else:
            new_str.append(temp)
    return " ".join(new_str)

#remove punctuation
def remove_punctuation(text):
    result = text.translate(str.maketrans('', '', string.punctuation))
    return result

#remove stopword
def remove_stopword(text):
    stop_words = set(stopwords.word("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word in stop_words]
    return filtered_text

#stemming
stemmer = PorterStemmer()
def stemming(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems

#lemmatizing
lemmatizer = WordNetLemmatizer()
def lemmatizing(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word) for word in word_tokens]
    return lemmas

#Part of Speech tagging POS
def pos_tagging(text):
    word_tokens = word_tokenize(text)
    tags = pos_tag(word_tokens)
    return tags

#Chunking
def chunking(text, grammar):
    word_tokens = word_tokenize(text)
    tags = pos_tag(word_tokens)
    chunkParser = nltk.RegexpParser(grammar)
    tree = chunkParser.parse(tags)
    for subree in tree.subtrees():
        print(subree)
    return tree.draw()

#Name entity recognition NER
def name_entity_recognition(text):
    word_tokens = word_tokenize(text)
    entities = ne_chunk(pos_tag(word_tokens))
    return entities
