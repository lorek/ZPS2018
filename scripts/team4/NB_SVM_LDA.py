import numpy
import pickle
import sys
import glob, os
import argparse

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer


def ParseArguments():
    parser = argparse.ArgumentParser(description="Project")

    parser.add_argument('--dataset',
                        default='flowers',
                        required=False,
                        help='Title of the dataset')
    parser.add_argument('--data-dir',
                        default="/Users/estera/repos/ZPS2018/BoW/flowers/sifts/kmeans50/hist_normalized/",
                        required=False,
                        help='Bag of Words directory')
    parser.add_argument('--result-dir',
                        default="/Users/estera/repos/ZPS2018/classify/flowers/sifts/kmeans50/hist_normalized/",
                        required=False,
                        help='Output classifier directory')
    parser.add_argument('--classifier-type',
                        default="SVM",
                        required=False,
                        help='Classifier type (SVM/LDA/NB)')

    args = parser.parse_args()

    return args.dataset, args.data_dir, args.result_dir, args.classifier_type

dataset, data_dir, result_dir, classifier_type = ParseArguments()

classes = []

for file in glob.glob(data_dir + "train/**"):
    print(file)
    #tmp = file.split('\\') #windows
    tmp=file.split('/') #linux/mac
    print(tmp)
    classes.append(tmp[len(tmp) - 1][:-7])

print("Klasy  = ", classes)

dictionaries = []
labels = []

for classs in classes:
    in_obj = open(data_dir + 'train/' + classs + '.pickle', 'rb')
    in_obj = pickle.load(in_obj)

    for i in range(len(in_obj) - 1):
        dictionaries.append(in_obj[i][1])
        labels.append(classs)

#print(labels)
#print(dictionaries)
#print(len(dictionaries))

v = DictVectorizer()
training_matrix = v.fit_transform(dictionaries)


if classifier_type == 'LDA': #to jeszcze nie działa
    print('LDA classifier chosen')

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(training_matrix, labels)

elif classifier_type == 'SVM':
    print('SVM classifier chosen')

    classifier = svm.SVC(gamma='scale')
    classifier.fit(training_matrix, labels)

elif classifier_type == 'NB':#to też niestety :<
    print('Naive Bayes classifier chosen')

    classifier = GaussianNB()
    classifier.fit(training_matrix, labels)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)


out_file = open(result_dir +  classifier_type + "/" + dataset + '_' + classifier_type + '.pickle', 'wb')
pickle.dump(classifier, out_file)