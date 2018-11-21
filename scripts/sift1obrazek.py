
import cv2
import numpy as np
import sys
import argparse
import json
from json_tricks import dump, dumps, load, loads, strip_comments
import glob, os


def ParseArguments():
	parser = argparse.ArgumentParser(description="Project ")
	parser.add_argument('--data-dir', default="", required=True, help='data dir')
	 
	args = parser.parse_args()
	
	return args.data_dir


# main progam
data_dir =  ParseArguments()

print("data-dir = ", data_dir)
 


# katalog data_dir/train ma miec podkatalogi = klasy

# najpierw odczytajmy tylko nazwy katalogow w data_dir/train i
 
classes=[]

for file in glob.glob(data_dir+"/train/**"): 
		tmp=file.split('\\')
		print(tmp)
		classes.append(tmp[len(tmp)-1])


print("Klasy  = ", classes)		


# teraz wykrywamy na kazdym obrazku zarowno z train jak i validate wszystkie sifty...
sifts_all=[]

counter_tmp = 1;


for classs in classes:
	#szukamy wszystkich plikow w data-dir/train a potem z data-dir/validate, zmienić poniżej
	for file in list((glob.glob(data_dir+"/validate/"+classs+"/**"))): 
		img = cv2.imread(file)
		sift = cv2.xfeatures2d.SIFT_create()
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		kp, sifts =  sift.detectAndCompute(gray,None)
		print("klasa = ", classs, ", obrazek = ",file, ", SIFTow: ", sifts.shape[0])
		s=str(counter_tmp)
		with open(classs+s+".json" , 'w') as ofile:
				json.dump(dumps(sifts),ofile)
				counter_tmp=counter_tmp+1
	counter_tmp=1
	

print("koniec")
