"""
This script contains the main sms classification code. We first load the processed
messages, convert each message into a bag of words and then into a tfidf vector
representation. Then we split the tfidf feature vector into training and test sets,
build our classifier on the training set, and test it on the test set.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
import matplotlib
# import matplotlib.pyplot as plt
import text_preprocessing

matplotlib.use('TkAgg')
pd.set_option('display.max_colwidth', None)


def tfidf_vectorization(messages):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    '''

    bow_transformer = CountVectorizer(
        analyzer=text_preprocessing.text_process
    ).fit(messages['message'])
    bow = bow_transformer.transform(messages['message']) # bag of words

    tfidf_transformer = TfidfTransformer().fit(bow)
    tfidf_vect = tfidf_transformer.transform(bow) # tfidf vector representation

    # store tfidf vector in a pickle file so tha it can be be used later in other scripts
    pickle.dump(tfidf_vect, open("output/tfidf_vector.pickle", "wb"))

    return tfidf_vect

def my_train_test_split(*datasets):
    '''
    Split dataset into training and test sets. We use a 70/30 split.
    '''
    return train_test_split(*datasets, test_size=0.3, random_state=101)

def train_classifier(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)

def predict_labels(classifier, X_test):
    return classifier.predict(X_test)

def main():

    messages = pd.read_csv('output/processed_msgs.csv')

    tfidf_vect = tfidf_vectorization(messages)

    # append our message length feature to the tfidf vector
    # to produce the final feature vector we fit into our classifiers
    len_feature = messages['length'].to_numpy()
    feat_vect = np.hstack((tfidf_vect.todense(), len_feature[:, None]))

    (X_train, X_test,
     y_train, y_test,
     _, test_messages) = my_train_test_split(feat_vect,
                                             messages['label'],
                                             messages['message'])

    classifiers = {
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        # 'Multinomial NB': MultinomialNB(),
        # 'KNN': KNeighborsClassifier(),
        # 'Random Forest': RandomForestClassifier(),
        # 'AdaBoost': AdaBoostClassifier(),
        # 'Bagging Classifier': BaggingClassifier()
    }

    assert np.array_equal(test_messages, test_messages)
    pred_scores = dict()
    pred = dict()
    # save misclassified messages
    file = open('output/misclassified_msgs.txt', 'a', encoding='utf-8')
    for key, value in classifiers.items():
        train_classifier(value, X_train, y_train)
        pred[key] = predict_labels(value, X_test)
        pred_scores[key] = [accuracy_score(y_test, pred[key])]
        print('\n############### ' + key + ' ###############\n')
        print(classification_report(y_test, pred[key]))

        # write misclassified messages into a new text file
        file.write('\n#################### ' + key + ' ####################\n')
        file.write('\nMisclassified Spam:\n\n')
        for msg in test_messages[y_test < pred[key]]:
            file.write(msg)
            file.write('\n')
        file.write('\nMisclassified Ham:\n\n')
        for msg in test_messages[y_test > pred[key]]:
            file.write(msg)
            file.write('\n')
    file.close()

    print('\n############### Accuracy Scores ###############')
    accuracy = pd.DataFrame.from_dict(pred_scores, orient='index', columns=['Accuracy Rate'])
    print('\n')
    print(accuracy)
    print('\n')

    # #plot accuracy scores in a bar plot
    # accuracy.plot(kind =  'bar', ylim=(0.85,1.0), edgecolor='black', figsize=(10,5))
    # plt.ylabel('Accuracy Score')
    # plt.title('Distribution by Classifier')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.show()
    # sys.exit(0)

    # Store "best" classifier
    dump(classifiers['Decision Tree'], 'output/model.joblib')

if __name__ == "__main__":
    main()
