
import cv2
import numpy as np
import random
import sys
import argparse
import json
from json_tricks import dump, dumps, load, loads, strip_comments
import glob, os


def ParseArguments():
	parser = argparse.ArgumentParser(description="Project ")
	parser.add_argument('--data-dir', default="", required=True, help='data dir')
	parser.add_argument('--fraction', default="0.1", required=False, help='fraction of files to consider')
	parser.add_argument('--sifts-file', default="", required=True, help='save all sifts in this file')
 
	
# 28.11.2018: Pawel Lorek: dodalem opcje "--fraction"
	
	#parser.add_argument('--show', default='yes', required=False, help='show images?') 
	args = parser.parse_args()
	
	return args.data_dir, args.sifts_file, args.fraction


# main progam
data_dir, sifts_file, fraction  =  ParseArguments()

print("data-dir = ", data_dir)
 


# katalog data_dir/train ma miec podkatalogi = klasy

# najpierw odczytajmy tylko nazwy katalogow w data_dir/train i 
classes=[]

for file in glob.glob(data_dir+"/train/**"): 
		#P. Lorek: UWAGA: byc moze pod Windowsem trzeba ponizsza linie zamienic na:
		#tmp=file.split('\\')
		tmp=file.split('/')
		print(tmp)
		classes.append(tmp[len(tmp)-1])


print("Klasy  = ", classes)		


# teraz wykrywamy na kazdym obrazku zarowno z train jak i validate wszystkie sifty...
sifts_all=[]

counter_tmp = 0;


for classs in classes:
	list1=glob.glob(data_dir+"/train/"+classs+"/**");
	#PL: shuffle list1
	random.shuffle(list1)
	#PL: take first fraction*len of the list 
	print("list 1 = ", list1)
	list1=list1[0:max(1,round(float(fraction)*len(list1)))]
	print("list 1 b = ", list1)
	list2=glob.glob(data_dir+"/validate/"+classs+"/**");
	
	#szukamy wszystkich plikow w data-dir/train oraz data-dir/validate
	#for file in list(np.concatenate((glob.glob(data_dir+"/train/"+classs+"/**"),glob.glob(data_dir+"/validate/"+classs+"/**")))): 
	for file in list(np.concatenate((list1,list2))): 
		img = cv2.imread(file)
		sift = cv2.xfeatures2d.SIFT_create()
		gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		kp, sifts =  sift.detectAndCompute(gray,None)
		print("klasa = ", classs, ", obrazek = ",file, ", SIFTow: ", sifts.shape[0])

		
		if(counter_tmp==0):
			sifts_all=sifts;
		else:
			sifts_all=np.append(sifts_all,sifts, axis=0)
		counter_tmp = counter_tmp+1


print(sifts_all.shape)
print("W sumie mamy SIFTow : ", sifts_all.shape[0])

with open(sifts_file , 'w') as outfile:
	#print(dumps(sifts_all))
	#json.dump(sifts_all,outfile)
	json.dump(dumps(sifts_all),outfile)
print("koniec")



		

 
