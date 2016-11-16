#!/usr/bin/env python
import os
from keras.preprocessing import image
import numpy as np
import deepdish as dd
"""
Temporary Monkey Patch
"""
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
tf.python.control_flow_ops = control_flow_ops



"""
For all valid image files within the INPUT_DIR_PATH,
a corresponding h5 file is created in the output folder containing
the bottleneck fingerprints of different pretrained models.
"""
INPUT_DIR_PATH = "/mount/SDB/paper-data/color"
OUTPUT_DIR_PATH = "/mount/SDB/paper-data/output"

MODES = ["InceptionV3", "VGG16", "VGG19","ResNet50"]
BATCH_SIZE = {
    "InceptionV3" : 128,
    "VGG16" : 128,
    "VGG19" : 128,
    "ResNet50" : 128,
    "Xception" : 128
}
TARGET_SIZE = {
    "InceptionV3" : (224, 224),
    "VGG16" : (224, 224),
    "VGG19" : (224, 224),
    "ResNet50" : (224, 224),
    "Xception" : (224, 224),
}
BUFFER = []
BUFFER_FILENAMES = []

for _MODE in MODES:
    if _MODE == "InceptionV3":
        from keras.applications.inception_v3 import InceptionV3 as _BASE_MODEL
        from keras.applications.inception_v3 import preprocess_input
        print "Initializing InceptionV3"
    elif _MODE == "VGG16":
        from keras.applications.vgg16 import VGG16 as _BASE_MODEL
        from keras.applications.vgg16 import preprocess_input
        print "Initializing VGG16"
    elif _MODE == "VGG19":
        from keras.applications.vgg19 import VGG16 as _BASE_MODEL
        from keras.applications.vgg19 import preprocess_input
        print "Initializing VGG19"
    elif _MODE == "ResNet50":
        from keras.applications.resnet50 import ResNet50 as _BASE_MODEL
        from keras.applications.resnet50 import preprocess_input
        print "Initializing ResNet50"
    elif _MODE == "Xception":
        from keras.applications.xception import Xception as _BASE_MODEL
        from keras.applications.Xception import preprocess_input
        print "Initializing Xception"
    else:
        continue

    def process_buffer():
        global TARGET_SIZE
        global BATCH_SIZE
        global OUTPUT_DIR_PATH
        global BUFFER
        global BUFFER_FILENAMES
        global total_processed

        for _idx, _image in enumerate(BUFFER):
            print "Processing : ", _image
            img = image.load_img(_image,
                                target_size=TARGET_SIZE[_MODE])
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            BUFFER[_idx] = x.reshape(x.shape[1:])
        print "Predicting %d images....." % len(BUFFER)
        RESULTS = base_model.predict(np.array(BUFFER))
        # old_shape = RESULTS.shape
        # flattened_shape = (RESULTS.shape[0], np.prod(RESULTS.shape[1:]))
        # RESULTS = RESULTS.reshape(flattened_shape)
        print "Shape : ",RESULTS.shape

        ##Write data in the corresponding output folders
        print "Writing the bottleneck fingerprints....."
        for _idx, _file in enumerate(BUFFER_FILENAMES):
            #Check if the target directory exists, if not create it
            target_folder = os.path.join(OUTPUT_DIR_PATH, _file[0].split("/")[-1], _file[1])
            if not (os.path.exists(target_folder) and os.path.isdir(target_folder)):
                os.makedirs(target_folder)

            dd.io.save(os.path.join(target_folder, _MODE+".h5"), RESULTS[_idx])

        BUFFER = []
        BUFFER_FILENAMES = []

    base_model = _BASE_MODEL(weights='imagenet', include_top=False)
    total_processed = 0

    for root, dirs, files in os.walk(INPUT_DIR_PATH, topdown=False):
        for name in files:
            BUFFER.append(os.path.join(root, name))
            BUFFER_FILENAMES.append((root, name))
            total_processed += 1
            print "Total Processed : ", total_processed
            if len(BUFFER) >= BATCH_SIZE[_MODE]:
                process_buffer()
    process_buffer()
