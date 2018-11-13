""" TODO:
    - tworzenie struktury katalogow i zapisywanie, gdzie trzeba
    - obsluga knn
    - stale z kodu...
"""


import cv2
import numpy as np
from scipy.cluster.vq import kmeans, whiten
import sys
import argparse
import json
from json_tricks import dump, dumps, load, loads, strip_comments
import glob, os

ALLOWED_ALGOS = ["k-means"]
SUBSETS = ["train", "validate"]
INFI = 2000000000

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--features-dir',
                            default = "features/cats_dogs/sifts/",
                            required = False,
                            help = 'data set directory name (features/cats_dogs/sifts/)')
    parser.add_argument('--algorithm',
                            default = "k-means",
                            required = False,
                            help='clustering alg. One of [k-means] (def. k-means)')
    parser.add_argument('--k',
                            default = 100,
                            required = False,
                            help = "depends on used algorithm. In k-means it's number of centers")
    parser.add_argument('--knn',
                            default = 0,
                            required = False,
                            help = "number of nearest neighbours in KNN. If 0, then we take the closest center")


    args = parser.parse_args()

    return args.features_dir, args.algorithm, args.k, args.knn

def load_features(features_dir):
    feats_per_utt = {}
    for subs in SUBSETS:
        for utt in glob.glob(features_dir + "/" + subs + "/*/*.json"):
            feats_file = open(utt)
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

feats_dir, algorithm, k_par, knn  =  ParseArguments()

if algorithm not in ALLOWED_ALGOS:
    print("Algorithm unknown.")
    exit(2)

cluster = None

feats_file = open(feats_dir + "sifts_all.json", "r")
all_feats = loads(load(feats_file, preserve_order = True))
feats_file.close()

if algorithm == "k-means":
    whitened_feats = whiten(all_feats) # numpy's k-means requires it

    cluster = kmeans(whitened_feats, k_par)

if cluster is not None:
    out_file = open("dict/cats_dogs/sifts/kmeans" + str(k_par) + "/cluster.json", 'w')
    json.dump(dumps(cluster), out_file)
    out_file.close()

feats_per_utt = load_features(feats_dir)

for utt in feats_per_utt.keys():
    splitted = utt.split("/")
    new_path = "dict/" + "/".join(splitted[1:3]) + "/kmeans" + str(k_par) + "/knn0/" + "/".join(splitted[4:]) + ".txt"

    print(new_path)

    out_clustered = open(new_path, 'w')
    for feat in feats_per_utt[utt]:
        out_clustered.write(str(find_class(feat, cluster, knn)) + "\n")
    out_clustered.close()
