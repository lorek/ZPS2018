""" TODO:
    - obsluga knn (kdt potrzebne, bo nigdy sie nie doliczy)
    - przejscie na sklearna, bo tam pewnie to wszystko jest
"""

import cv2
import numpy as np
from scipy.cluster.vq import kmeans, whiten
import sys
import argparse
import json
from json_tricks import dump, dumps, load, loads, strip_comments
import glob, os

ALLOWED_ALGOS = ["kmeans"]
SUBSETS = ["train", "validate"]
INFI = 2000000000

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--features-dir',
                            default = "features/cats_dogs/sifts/",
                            required = False,
                            help = 'data set directory name (features/cats_dogs/sifts/)')
    parser.add_argument('--algorithm',
                            default = "kmeans",
                            required = False,
                            help='clustering alg. One of [kmeans] (def. kmeans)')
    parser.add_argument('--k',
                            default = 100,
                            required = False,
                            help = "depends on used algorithm. In k-means it's number of centers")
    parser.add_argument('--knn',
                            default = 0,
                            required = False,
                            help = "number of nearest neighbours in KNN. If 0, then we take the closest center")


    args = parser.parse_args()

    return args.features_dir, args.algorithm, int(args.k), int(args.knn)

def load_features(features_dir):
    feats_per_utt = {}
    for subs in SUBSETS:
        for utt in glob.glob(features_dir + "/" + subs + "/*/*.json"):
            try:
                feats_file = open(utt)
            except:
                print("Couldn't open {}.".format(utt))
                exit(4)
            feats_per_utt[utt] = loads(load(feats_file))
            feats_file.close()

    return feats_per_utt

def dist(x, y):
    return sum((x-y) ** 2)

def find_class(feat, cluster, knn):
    knn = 0 # TODO

    if(knn == 0):
        best_center = -1
        best_dist = INFI
        for c in range(0, cluster[0].shape[0]):
            new_dist = dist(cluster[0][c], feat)
            if new_dist < best_dist:
                best_dist = new_dist
                best_center = c

    return best_center

"""
    Main part
"""

# Process args

feats_dir, algorithm, k_par, knn  =  ParseArguments()

if algorithm not in ALLOWED_ALGOS:
    print("Algorithm {} not allowed. Try one of {}".format(algorithm, ALLOWED_ALGOS))
    exit(2)

# Load all features

try:
    feats_path = feats_dir + "sifts_all.json"
    feats_file = open(feats_path, "r")
except:
    print("Couldn't open features file: {}.".format(feats_path))
    exit(3)

all_feats = loads(load(feats_file, preserve_order = True))
feats_file.close()

# Train and save cluster

cluster = None
params_str = ""

if algorithm == "kmeans":
    whitened_feats = whiten(all_feats) # numpy's k-means requires it
    cluster = kmeans(whitened_feats, k_par)
    params_str += str(k_par)

if cluster is not None:
    cluster_path = "dict/" + "/".join(feats_dir.split("/")[1:3]) + "/" + algorithm + params_str + "/"

    if not os.path.exists(cluster_path):
        os.makedirs(cluster_path)

    out_file = open(cluster_path + "cluster.json", 'w')
    json.dump(dumps(cluster), out_file)
    out_file.close()

# Load features per utterance (image)

feats_per_utt = load_features(feats_dir)

for utt in feats_per_utt.keys():
    utt_splt = utt.split("/")
    no_ext = ".".join(utt_splt[-1].split(".")[:-1])
    utt_path = cluster_path + "/knn" + str(knn) + "/" + "/".join(utt_splt[4:-1]) + "/"

    print(utt_path)

    if not os.path.exists(utt_path):
        os.makedirs(utt_path)

    out_clustered = open(utt_path + no_ext + ".txt", 'w')
    for feat in feats_per_utt[utt]:
        out_clustered.write(str(find_class(feat, cluster, knn)) + "\n")
    out_clustered.close()
