import numpy
import pickle
import sys
import glob, os
import argparse

from sklearn.naive_bayes import GaussianNB
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
    parser.add_argument('--classifier-dir',
                        default="/Users/estera/repos/ZPS2018/classify/flowers/sifts/kmeans50/hist_normalized/",
                        required=False,
                        help='Classifier directory')
    parser.add_argument('--classifier-type',
                        default="NB",
                        required=False,
                        help='Classifier type (SVM/LDA/NB)')

    args = parser.parse_args()

    return args.dataset, args.data_dir, args.classifier_dir, args.classifier_type

dataset, data_dir, classifier_dir, classifier_type = ParseArguments()


obj = open(classifier_dir + classifier_type + "/" + dataset + "_" + classifier_type + '.pickle', 'rb')
classifier = pickle.load(obj)

classes = []

for file in glob.glob(data_dir + "validate/**"):
    #print(file)
    #tmp = file.split('\\') #windows
    tmp=file.split('/') #linux/mac
    #print(tmp)
    classes.append(tmp[len(tmp) - 1][:-7])

print("Klasy  = ", classes)

dictionaries = []
labels = []
n=[0]

for classs in classes:
    in_obj = open(data_dir + 'validate/' + classs + '.pickle', 'rb')
    in_obj = pickle.load(in_obj)

    for i in range(len(in_obj) - 1):
        dictionaries.append(in_obj[i][1])
        labels.append(classs)

    n.append(len(labels))

#print(labels)
#print(dictionaries)
#print(len(dictionaries))

v = DictVectorizer()
matrix = v.fit_transform(dictionaries)


if classifier_type == 'LDA':
    print('LDA classifier chosen to check on '+ dataset)

    a = classifier.predict(matrix)

elif classifier_type == 'SVM':
    print('SVM classifier chosen to check on '+ dataset)

    a = classifier.predict(matrix)

elif classifier_type == 'NB':
    print('Naive Bayes classifier chosen to check on '+ dataset)

    a = classifier.predict(matrix.todense())

#if not os.path.exists(result_dir):
#    os.makedirs(result_dir)


#v = DictVectorizer()
#matrix = v.fit_transform(dictionaries)
#print(matrix)


#a = classifier.predict(matrix.todense())

for i in range(len(n)-1):
    n1=n[i]
    n2=n[i+1]

    k = (labels[n1:n2] == a[n1:n2]).sum()
    p = k / (n2-n1)
    # print(a)
    print(classes[i])
    print(p)



#print(n)
k=(labels == a).sum()
p=k/n[len(n)-1]
#print(a)
print("Total")
print(p)
#print(len(n)==len(classes))