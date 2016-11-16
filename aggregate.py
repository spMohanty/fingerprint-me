#!/usr/bin/env python

import numpy as np
from PIL import Image
import deepdish as dd
import glob
import random

INPUT_DIR_PATH = "/mount/SDC/paper-data/output"
AGGREGATED_OUTPUT_PATH = "/mount/SDC/paper-data/output-aggregated"

SIZE = (256, 256)
labels_dict = {}
label_count = 0

def preprocess(inputdir, outputdir, train_percent, isSmall):
    print "Input Dir : ", inputdir
    print "Output Dir : ", outputdir

    def generate_object():
        global labels_dict
        global label_count
        IMAGES = []
        COARSE_LABELS = []
        DENSE_LABELS = []
        FILENAMES = []
        #Iterate over the Input Directory's train/validate folder to collect the images
        #target_files = glob.glob(inputdir+"/*/*/ResNet50.h5")
        target_files = glob.glob(inputdir+"/*/*/InceptionV3.h5")
        random.shuffle(target_files)
        for idx, _file in enumerate(target_files):
            print _file
            data = dd.io.load(_file)
            data = np.mean(data, axis=(0,1)) # Leading to (5,5, 2048) -> (2048, 1)
            _label = _file.split("/")[-3]
            try:
                foo = labels_dict[_label]
            except:
                label_count += 1
                labels_dict[_label] = label_count

            IMAGES.append(data)
            DENSE_LABELS.append(_label)
            COARSE_LABELS.append(labels_dict[_label])
            FILENAMES.append(_file.split("/")[-1])
            #Option to create small version of the dataset for debugging
            if isSmall:
                if idx > 5000:
                    break

        train_length = int(train_percent*len(IMAGES))
        TRAIN_DATA = {'data' : np.array(IMAGES)[:train_length], 'coarse_labels': np.array(COARSE_LABELS)[:train_length], 'dense_labels': np.array(DENSE_LABELS)[:train_length], 'filenames': np.array(FILENAMES)[:train_length]}
        VALIDATION_DATA = {'data' : np.array(IMAGES)[train_length:], 'coarse_labels': np.array(COARSE_LABELS)[train_length:], 'dense_labels': np.array(DENSE_LABELS)[train_length:], 'filenames': np.array(FILENAMES)[train_length:]}
        return TRAIN_DATA, VALIDATION_DATA

    TRAIN_DATA, VALIDATION_DATA  = generate_object()
    if isSmall:
        dd.io.save(outputdir+"/train-small.h5", TRAIN_DATA)
        dd.io.save(outputdir+"/validation-small.h5", VALIDATION_DATA)
    else:
        dd.io.save(outputdir+"/train.h5", TRAIN_DATA)
        dd.io.save(outputdir+"/validation.h5", VALIDATION_DATA)
    dd.io.save(outputdir+"/labels-dict.h5", labels_dict)

import sys, getopt

def main(argv):
   inputdir = None
   outputdir = None
   isSmall = False
   train_percent = 0.8
   # try:
   #    opts, args = getopt.getopt(argv,"hd:o:",["input_directory=", "output_directory="])
   # except getopt.GetoptError:
   #    print 'preprocess.py -d <path_to_plantvillage_directory> -o <output_directory>'
   #    sys.exit(2)
   # for opt, arg in opts:
   #    if opt == '-h':
   #       print 'preprocess.py -d <path_to_plantvillage_directory> -o <output_directory>'
   #       sys.exit()
   #    elif opt in ("-d", "--input_directory"):
   #       inputdir = arg
   #    elif opt in ("-o", "--output_directory"):
   #       outputdir = arg
   #    elif opt in ("-s", "--small"):
   #       isSmall = True
   #
   # if inputdir == None or outputdir == None:
   #       print 'Usage: preprocess.py -d <path_to_plantvillage_directory> -o <output_directory>'
   #       sys.exit(2)

   # preprocess(inputdir, outputdir, isSmall)
   preprocess(INPUT_DIR_PATH, AGGREGATED_OUTPUT_PATH, 0.8, isSmall)

if __name__ == "__main__":
   main(sys.argv[1:])
