import numpy as np
import pickle
import argparse

from sklearn.feature_extraction import DictVectorizer
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    
    parser.add_argument('--dataset',
                        default='',
                        required=True,
                        help='Title of the dataset')
    parser.add_argument('--data-dir',
                        default="",
                        required=True,
                        help='Bag of words directory')
    parser.add_argument('--classifier-dir',
                        default="",
                        required=True,
                        help='Classifier directory')
    parser.add_argument('--classifier-type',
                        default="",
                        required=True,
                        help="Classifier type (decision-tree / random-forest / svm)")
    parser.add_argument('--use-tfidf',
                        default='F',
                        required=False,
                        help='Use TF-IDF instead of histograms (T/F)')
    
    args = parser.parse_args()

    return args.dataset, args.data_dir, args.classifier_dir, args.classifier_type, args.use_tfidf

'''
    Main part
'''

dataset, data_dir, classifier_dir, classifier_type, use_tfidf = ParseArguments()

# Append set name to data directory

data_dir = data_dir + dataset + '/'

# Set classifier path and name

full_path = classifier_dir + dataset + '/' + dataset + '_' + classifier_type

if use_tfidf == 'T':
    full_path += '_tfidf'

full_path += '.pickle'

# Load and check

in_obj = open(full_path, 'rb')
my_classifier = pickle.load(in_obj)

all_true = []
all_pred = []

for classs in my_classifier.classes_:
    
    #print(classs)
    in_obj = open(data_dir + 'validate/' + classs + '.pickle', 'rb')
    
    test_dicts = pickle.load(in_obj)
    test_size = len(test_dicts)-1

    test_dicts = [test_dicts[len(test_dicts)-1][i] for i in range(len(test_dicts)-1)]
    #print(test_dicts[1])
    
    matrix = DictVectorizer()
    X = matrix.fit_transform(test_dicts)

    predictions = my_classifier.predict(X)
    
    all_pred = np.append(all_pred, predictions)
    all_true = np.append(all_true, [classs] * test_size)
    
    print('Accuracy for', classs, ':', accuracy_score([classs] * test_size, predictions))

print(accuracy_score(all_true, all_pred))