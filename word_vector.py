import spacy
import numpy as np

nlp = spacy.load('en_core_web_lg')

text = "These vectors can be used as features for machine learning models."
with nlp.disable_pipes():
    vectors = np.array([token.vector for token in nlp(text)])
vectors.shape

