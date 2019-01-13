import numpy
import pickle
import sys
import glob, os
import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    
    parser.add_argument('--dataset',
                        default='',
                        required=True,
                        help='Title of the dataset')
    parser.add_argument('--data-dir',
                        default="",
                        required=True,
                        help='Bag of Words directory')
    parser.add_argument('--result-dir',
                        default="",
                        required=True,
                        help='Output classifier directory')
    parser.add_argument('--classifier-type', default="",
                        required=True,
                        help='Classifier type (decision-tree / random-forest)')
    parser.add_argument('--use-tfidf',
                        default='F',
                        required=False,
                        help='Use TF-IDF instead of histograms (T/F)')
    
    args = parser.parse_args()

    return args.dataset, args.data_dir, args.result_dir, args.classifier_type, args.use_tfidf

"""
    Main part
"""

dataset, data_dir, result_dir, classifier_type, use_tfidf = ParseArguments()
# Create list of classes

classes = []

for file in glob.glob(data_dir+"/train/**"):
        print(file)
        tmp=file.split('\\')
        #tmp=file.split('/')
        print(tmp)
        classes.append(tmp[len(tmp)-1][:-7])

print("Klasy  = ", classes)

dictionaries = []
labels = []

for classs in classes:
    in_obj = open(data_dir + 'train/' + classs + '.pickle', 'rb')
    in_obj = pickle.load(in_obj)
    
    if use_tfidf == 'F':
        for i in range(len(in_obj)-1):
            dictionaries.append(in_obj[i][1])
            labels.append(classs)
    else:
        for i in range(len(in_obj)-1):
            dictionaries.append(in_obj[len(in_obj)-1][i])
            labels.append(classs)

v = DictVectorizer()
training_matrix = v.fit_transform(dictionaries)

if classifier_type == 'decision-tree':
    print('Decision tree classifier chosen')
    
    classifier = DecisionTreeClassifier()
    classifier.fit(training_matrix, labels)
    
elif classifier_type == 'random-forest':
    print('Random forest classifier chosen')
    
    classifier = RandomForestClassifier()
    classifier.fit(training_matrix, labels)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
if tf_idf == 'F':
    out_file = open(result_dir + dataset + '_' + classifier_type + '.pickle', 'wb')
else:
    out_file = open(result_dir + dataset + '_' + classifier_type + 'tf_idf.pickle', 'wb')
pickle.dump(classifier, out_file)