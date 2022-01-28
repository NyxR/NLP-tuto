from ssl import wrap_socket
import nltk
import re
import string
import inflect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#Text preprocessing step-1

#to lowercase
def text_lowercase(text):
    return text.lower()

input_str = """Hey, did you know that the summer 
               break is coming? Amazing right !! 
               It's only 5 more days !!"""

print(text_lowercase(input_str))

#remove number
def remove_number(text):
    result = re.sub(r'\d+', '', text) #replace all digit in text to ''
    return result

#convert number into text
p = inflect.engine()
def convert_number(text):
    temp_str = text.split()
    new_str = []
    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_str.append(temp)
        else:
            new_str.append(word)
    temp_str = ' '.join(new_str)
    return temp_str

input_str = 'There are 3 balls in this bag, and 12 in the other one.'
print(convert_number(input_str))

#remove punctuation
def remove_punctuation(text):
    translator = str.maketrans('','', string.punctuation)
    result = text.translate(translator)
    return result

input_str = "Hey, did you know that the summer break is coming? Amazing right !! It's only 5 more days !!"
print(remove_punctuation(input_str))

#remove whitespace
def remove_whitespace(text):
    return " ".join(text.split())

input_str = "   we don't need   the given questions"
print(remove_whitespace(input_str))

#remove stopwords
def remove_stopwords(text):
    stop_words = set(stopwords.word("english")) #get all the english stopword list
    word_tokens = word_tokenize(text) #tokenize the text
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text

example_text = "This is a sample sentence and we are going to remove the stopwords from this."
print(remove_stopwords(example_text))

#Stemming
stemmer = PorterStemmer()
def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems

text = 'data science uses scientific methods algorithms and many types of processes'
print(stem_words(text))

#Lemmatisation
lematizer = WordNetLemmatizer()
def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    lemmas = [lematizer.lemmatize(word) for word in word_tokens]
    return lemmas

text = 'data science uses scientific methods algorithms and many types of processes'
print(lemmatize_word(text))


