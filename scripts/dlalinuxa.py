# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 23:41:50 2018

@author: Wojtek
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 18:09:43 2018

@author: czoppson
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:36:06 2018

@author: czoppson
"""
''' Mam dwa foldery koty i psy a w nich po kilka plikow tekstowych, ten kod przechodzi 
mi po nich i tworze pickle gdzie mam w master_list zapisene listy, 
[[slownik_zywukłyplik1,slownik_unormowanyplik1],[slownik_zwykłyplik2,...],[slownik_tfidf_folder]]
 
'''
import numpy as np
import sys
import argparse
import json
#from json_tricks import dump, dumps, load, loads, strip_comments
import glob, os
import pickle

def ParseArguments():
    parser = argparse.ArgumentParser(description = 'Bag of Words')
    parser.add_argument('--data-dir',
                        default = 'C:\\Users\\Wojtek\\repos\\ZPS2018\\dict\\animals\\sifts\\kmeans50\\knn0',
                        required = False,
                        help = 'path to folder with images')
    parser.add_argument('--result-dir',
                        default = 'F:\\STUDIA\\ZPS\\wyniki',
                        required = False,
                        help = 'path to folder with results')
    args = parser.parse_args()
    return args.data_dir, args.result_dir
data_dir , result_dir = ParseArguments()
print(data_dir)
classes = []
for file in glob.glob(data_dir+"/"+"train"+"/**"):
    tmp=file.rsplit('/')
    classes.append(tmp[len(tmp)-1])
print("Klasy  = ", classes)

for classs in classes:
    print(classs)
    for grupa in ['train','validate']:
        print(grupa)
        master_list = []
        matrix = []
        for filename in glob.glob(os.path.join(data_dir + '/'+ grupa + '/'+ classs + '/', '*.txt')):
            print(filename)
            wyniki = []
            with open(filename) as f:
                plik = f.readlines()
                dlugosc = len(plik)
                liczby = [i for i in range(1, dlugosc+1)]
                #print(liczby)
                
                sifty = []
                for sift in plik:
                    sift = sift.strip('\n')
                    sift = int(sift)
                    sifty.append(sift)
                
                # Dla macierzy czesc A
                suma_siftow = sum(sifty)
                sifty_znormalizowane = [round(x/suma_siftow,3) for x in sifty]
                matrix.append(sifty_znormalizowane)
                
                # Tutaj normlany slownik
                pary = zip(liczby, sifty)
                result = set(pary)
                slownik = dict(result)
             
                wyniki.append(slownik)
                
                #Tutaj unormowany slownik
                suma = sum(slownik.values())
                slownik_norma = dict(result)
                for k,v in slownik_norma.items():
                    slownik_norma[k] = v/suma
               
               
                wyniki.append(slownik_norma)
                master_list.append(wyniki)
                #print(master_list)
                
        A = np.transpose(np.array(matrix))
        zeros = []
        B = (A==0)
     
        for i in range(dlugosc):# to trzeba zmienic dla innego przypadku niz kmeans 50 
            zeros.append(sum(B[i][:]))# liczy ile w danym wierszu było zer
        
        zeros = np.array(zeros)
        N = [A.shape[1]]*A.shape[0] # lista dlugosci ilosci siftow o wartosciach rownych ilosci plikow
        N = np.array(N) # ilosc plikow razy ilosc siftow 
        idf = np.log(N/(N-zeros))
        tf_idf = np.zeros((A.shape[0],A.shape[1]))
        for i in range(A.shape[0]):
            tf_idf[i,:] = A[i][:]*idf[i]
        
        dict_list = []
        do_zipa  = list(range(1,dlugosc+1))
        for i in range(tf_idf.shape[1]):
            pary = zip(do_zipa,tf_idf[:,i])
            result = set(pary)
            dict_list.append(dict(result))
            
        master_list.append(dict_list)
        a = master_list
        if not os.path.exists(result_dir + '/' + grupa):
            os.makedirs(result_dir + '/' + grupa)
        pickle_out = open(result_dir + '/' + grupa + '/' + classs +  '.pickle','wb')
        pickle.dump(a,pickle_out)
        pickle_out.close()
                #print(len(master_list))
    