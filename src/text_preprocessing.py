"""
Preprocess the data to be trained by the learning algorithm.
"""

import pandas as pd
import numpy as np

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from joblib import dump, load

def _load_data():
    messages = pd.read_csv(
        'smsspamcollection/SMSSpamCollection',
        sep='\t',
        names=['label', 'message']
    )
    return messages

def _text_process(data):
    '''
    1. remove punc
    2. do stemming of words
    3. remove stop words
    4. return list of clean text words
    '''
    nopunc = [c for c in data if c not in string.punctuation] #remove punctuations
    nopunc = ''.join(nopunc)

    stemmed = ''
    nopunc = nopunc.split()
    for i in nopunc:
        stemmer = SnowballStemmer('english')
        stemmed += (stemmer.stem(i)) + ' ' # stemming of words

    clean_msgs = [
        word for word in stemmed.split()
        if word.lower() not in stopwords.words('english')
    ] # remove stopwords

    return clean_msgs

def _extract_message_len(data):
    # return as np.array and reshape so that it works with make_union
    return np.array([len(message) for message in data]).reshape(-1, 1)

def _preprocess(messages):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    3. Add message length
    '''
    preprocessor = make_union(
        make_pipeline(
            CountVectorizer(analyzer=_text_process),
            TfidfTransformer()
        ),
        # append the message length feature to the vector
        FunctionTransformer(_extract_message_len, validate=False)
    )

    preprocessed_data = preprocessor.fit_transform(messages['message'])
    dump(preprocessor, 'output/preprocessor.joblib')
    dump(preprocessed_data, 'output/preprocessed_data.joblib')
    return preprocessed_data

def prepare(message):
    preprocessor = load('output/preprocessor.joblib')
    return preprocessor.transform([message])


def main():
    messages = _load_data()
    print('\n################### Processed Messages ###################\n')
    with pd.option_context('expand_frame_repr', False):
        print(messages)
    _preprocess(messages)

if __name__ == "__main__":
    main()
