import numpy
import pickle


from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer

obj = open('/Users/estera/repos/ZPS2018/classify/flowers/sifts/kmeans50/hist_normalized/SVM/flowers_SVM.pickle', 'rb')
classifier = pickle.load(obj)

obj = open('/Users/estera/repos/ZPS2018/BoW/flowers/sifts/kmeans50/hist_normalized/validate/daisy.pickle', 'rb')
obj = pickle.load(obj)
#print(len(obj))
dictionaries=[]
for i in range(len(obj) - 1):
    dictionaries.append(obj[i][1])

obj = open('/Users/estera/repos/ZPS2018/BoW/flowers/sifts/kmeans50/hist_normalized/validate/dandelion.pickle', 'rb')
obj = pickle.load(obj)
#print(len(obj))
for i in range(len(obj) - 1):
    dictionaries.append(obj[i][1])

v = DictVectorizer()
matrix = v.fit_transform(dictionaries)
#print(matrix)

a = classifier.predict(matrix)
print(a)