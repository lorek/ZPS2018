 
 

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
	parser.add_argument('--sifts-file', default="", required=True, help='save all sifts in this file')
 
	
	
	#parser.add_argument('--show', default='yes', required=False, help='show images?') 
	args = parser.parse_args()
	
	return args.data_dir, args.sifts_file


# main progam
data_dir, sifts_file  =  ParseArguments()

print("data-dir = ", data_dir)
 


# katalog data_dir/train ma miec podkatalogi = klasy

# najpierw odczytajmy tylko nazwy katalogow w data_dir/train i 
classes=[]

for file in glob.glob(data_dir+"/train/**"): 
		tmp=file.rsplit('/',3)		
		classes.append(tmp[len(tmp)-1])

print("Klasy  = ", classes)		

# teraz przeczytajmy wszystkie pliki w train (tutaj je tylko wyswietlamy




# teraz wykrywamy na kazdym obrazku zarowno z train jak i validate wszystkie sifty...
sifts_all=[]

counter_tmp = 0;


for classs in classes:
	#szukamy wszystkich plikow w data-dir/train oraz data-dir/validate
	for file in list(np.concatenate((glob.glob(data_dir+"/train/"+classs+"/**"),glob.glob(data_dir+"/validate/"+classs+"/**")))): 
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
		#tutaj mozna wczytac obrazek, szukac SIFTow... itd., itp.


print(sifts_all.shape)
print("W sumie mamy SIFTow : ", sifts_all.shape[0])

with open(sifts_file , 'w') as outfile:
	#print(dumps(sifts_all))
	#json.dump(sifts_all,outfile)
	json.dump(dumps(sifts_all),outfile)



# tak bysmy wczytali plik (sifts_all_loaded bedzie macierza NumPy):
#with open(sifts_file, 'r') as infile:
#	sifts_all_loaded = loads(load(infile, preserve_order=True))
#
#  print("pp ", sifts_all_loaded)
		
		 

		

 
