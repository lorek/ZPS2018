import cv2
import numpy as np
import sys
import argparse
import glob, os
import pickle

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--data-dir', default="", required=True, help='data dir')
    parser.add_argument('--sift-dir', default="", required=True, help='sift dir') 
    args = parser.parse_args()

    return args.data_dir, args.sift_dir

# main progam
data_dir,sift_dir =  ParseArguments()

print("data-dir = ", data_dir)

# katalog data_dir/train ma miec podkatalogi = klasy

# najpierw odczytajmy tylko nazwy katalogow w data_dir/train
 
classes=[]

for file in glob.glob(data_dir+"/train/**"): 
        #P. Lorek: UWAGA: byc moze pod Windowsem trzeba ponizsza linie zamienic na:
        #tmp=file.split('\\')
        tmp=file.split('/')
        print(tmp)
        classes.append(tmp[len(tmp)-1])


print("Klasy  = ", classes)

# teraz wykrywamy na kazdym obrazku zarowno z train jak i validate wszystkie sifty

#szukamy wszystkich plikow w data-dir/train a potem z data-dir/validate

for classs in classes:
    if not os.path.exists(sift_dir+"/train/"+classs):
        os.makedirs(sift_dir+"/train/"+classs)

    sifts_per_utt = {}

    for file in list((glob.glob(data_dir+"/train/"+classs+"/**"))):
        img = cv2.imread(file)
        sift = cv2.xfeatures2d.SIFT_create()
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, sifts =  sift.detectAndCompute(gray,None)
        sifts_per_utt[file] = sifts
        print("klasa = ", classs, ", obrazek = ",file, ", SIFTow: ", sifts.shape[0])

    with open(sift_dir+ "/train/" + classs + "/feats.pickle" , 'wb') as ofile:
        pickle.dump(sifts_per_utt,ofile)

for classs in classes:
    if not os.path.exists(sift_dir+"/validate/"+classs):
        os.makedirs(sift_dir+"/validate/"+classs)

    sifts_per_utt = {}

    for file in list((glob.glob(data_dir+"/validate/"+classs+"/**"))):
        img = cv2.imread(file)
        sift = cv2.xfeatures2d.SIFT_create()
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp, sifts =  sift.detectAndCompute(gray,None)
        sifts_per_utt[file] = sifts
        print("klasa = ", classs, ", obrazek = ",file, ", SIFTow: ", sifts.shape[0])

    with open(sift_dir+ "/validate/" + classs + "/feats.pickle" , 'wb') as ofile:
        pickle.dump(sifts_per_utt,ofile)

print("koniec")
