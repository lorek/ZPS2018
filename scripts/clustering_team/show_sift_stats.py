""" TODO:
    - obsluga knn (kdt potrzebne, bo nigdy sie nie doliczy)
    - przejscie na sklearna, bo tam pewnie to wszystko jest
"""

import cv2
import numpy as np
from scipy.cluster.vq import kmeans, whiten
import sys
import argparse
import glob, os
import pickle

 

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--file',
                            required = False,
                            )

    args = parser.parse_args()

    return args.file

def load_class_features(class_path):
    pickle_path = class_path + "/feats.pickle"

    try:
        feats_file = open(pickle_path, 'rb')
    except:
        print("Couldn't open {}.".format(pickle_path))
        exit(5)

    feats_per_utt = pickle.load(feats_file)

    return feats_per_utt

"""
    Main part
"""

# Process args

plik  =  ParseArguments()



try:
    feats_file = open(plik, "rb")
except:
    print("Couldn't open features file: {}.".format(feats_path))
    exit(3)


all_feats = pickle.load(feats_file)
feats_file.close()

for feat in all_feats:
    print(feat)
    print(all_feats[feat])
    

# Train and save cluster

# ~ for subs in SUBSETS:
    # ~ all_classess = glob.glob(feats_dir + "/" + subs + "/*")

    # ~ for class_path in all_classess:
        # ~ feats_per_utt = load_class_features(class_path)

        # ~ for utt in feats_per_utt.keys():
            # ~ utt_splt = list(filter(None, utt.split("/"))) # remove empty strings (coz of ///// in paths)
            # ~ no_ext = ".".join(utt_splt[-1].split(".")[:-1])
            # ~ utt_path = cluster_path + "/knn" + str(knn) + "/" + "/".join(utt_splt[2:-1]) + "/"

            # ~ if not os.path.exists(utt_path):
                # ~ os.makedirs(utt_path)

            # ~ out_clustered = open(utt_path + no_ext + ".txt", 'w')

            # ~ hist = [0] * k_par

            # ~ for feat in feats_per_utt[utt]:
                # ~ hist[find_class(feat, cluster, knn)] += 1

            # ~ for i in range(k_par):
                # ~ out_clustered.write(str(hist[i]) + '\n')

            # ~ out_clustered.close()
