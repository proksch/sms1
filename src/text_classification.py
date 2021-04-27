"""
Train the model using different algorithms.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from text_preprocessing import _load_data

#matplotlib.use('TkAgg')
pd.set_option('display.max_colwidth', None)


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

    raw_data = _load_data()
    preprocessed_data = load('output/preprocessed_data.joblib')

    (X_train, X_test,
     y_train, y_test,
     _, test_messages) = my_train_test_split(preprocessed_data,
                                             raw_data['label'],
                                             raw_data['message'])

    classifiers = {
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Multinomial NB': MultinomialNB(),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging Classifier': BaggingClassifier()
    }

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

    #plot accuracy scores in a bar plot
    accuracy.plot(kind='bar', ylim=(0.85, 1.0), edgecolor='black', figsize=(10, 5))
    plt.ylabel('Accuracy Score')
    plt.title('Distribution by Classifier')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("output/accuracy_scores.png")

    # Store "best" classifier
    dump(classifiers['Decision Tree'], 'output/model.joblib')

if __name__ == "__main__":
    main()
