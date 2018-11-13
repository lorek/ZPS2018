import glob, os
import argparse
import random
from shutil import copyfile

def ParseArguments():
    parser = argparse.ArgumentParser(description="Project ")
    parser.add_argument('--data-set',
                            default = "",
                            required = True,
                            help = 'data set name')

    parser.add_argument('--percent',
                            default = "",
                            required = True,
                            help = 'percent to test set')

    args = parser.parse_args()

    return args.data_set, int(args.percent)

data_set, percent = ParseArguments()

utt_by_class = {}

for utt in glob.glob("datasets/" + data_set + "/*/*.jpg"):
    utt_splt = utt.split("/")
    class_name = utt_splt[-2]
    if class_name not in utt_by_class:
        utt_by_class[class_name] = []

    utt_by_class[class_name].append(utt)

for c in utt_by_class.keys():
    how_many = int(len(utt_by_class[c]) * percent / 100)
    to_test = random.sample(range(0, len(utt_by_class[c])), how_many)

    new_path_validate = "datasets/" + data_set + "/validate/" + str(c) + "/"
    new_path_train    = "datasets/" + data_set + "/train/" + str(c) + "/"

    if not os.path.exists(new_path_validate):
        os.makedirs(new_path_validate)
    if not os.path.exists(new_path_train):
        os.makedirs(new_path_train)

    for i in range(len(utt_by_class[c])):
        if i in to_test:
            copyfile(utt_by_class[c][i], "datasets/" + data_set + "/validate/" + str(c) + "/" + os.path.basename(utt_by_class[c][i]))
        else:
            copyfile(utt_by_class[c][i], "datasets/" + data_set + "/train/" + str(c) + "/" + os.path.basename(utt_by_class[c][i]))
