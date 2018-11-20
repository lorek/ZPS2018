#import cv2
import numpy as np
import sys
import argparse
import json
#from json_tricks import dump, dumps, load, loads, strip_comments
import glob, os
path = 'F:\STUDIA\ZPS'
result_path = 'F:\STUDIA\ZPS\wyniki'
#folder_lista = []
#ostateczna = []
methods = ['hist', 'hist_normalized', 'tfidf']
for method in methods:
    tmp = method
    print(tmp)
    for filename in glob.glob(os.path.join(path, '*.txt')):
        method = {}
        #print(filename)
        wartosci = []
        with open(filename) as f:
            plik = f.readlines()
        for sift in plik:
            sift = sift.strip('\n')
            sift = int(sift)
            if sift not in method.keys() and tmp == 'hist':
                method[sift] = 1
            if sift not in method.keys() and tmp == 'hist_normalized':
                method[sift] = 1/len(plik)
            if sift not in method.keys() and tmp == 'tfidf':
                pass
            if tmp == 'hist':
                method[sift] = method[sift] + 1
            if tmp == 'hist_normalized':
                method[sift] = method[sift] + 1/len(plik)
                #print(method[sift])
            if tmp == 'tfidf':
                pass
            #trzeba to zrobić
        print('METODA '+tmp+', WYNIK:')
        #print(str(method))
        #folder_lista.append(method)
        result_filename = filename.split('\\')[-1]
        result_filename = result_filename.strip('.txt')+ '_' + str(tmp) + '.txt'
        result = open(result_path + '\\'  + result_filename, 'w')
        result.write(str(method))
        result.close()

