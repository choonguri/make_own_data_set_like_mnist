from PIL import Image
import gzip
import pickle
from glob import glob
import numpy as np
import pandas as pd

TRAIN_SET_COUNT = 5
VALIDATION_SET_COUNT = 3
TEST_SET = 2
TARGET_IMAGES = "./images/*.jpg"
LABEL_FILEPATH = "./images/label.txt"
PKL_FILE = "dataset.pkl.gz"

def generate_dataset(glob_files, label_filepath=""):
    dataset = []
    for _, file_name in enumerate(sorted(glob(glob_files), key=len)):
        img = Image.open(file_name)
        pixels = [list(img.getdata())]
        dataset.append(pixels)
    if len(label_filepath) > 0:
        lb = pd.read_csv(label_filepath)
        return np.array(dataset), np.array(lb["class"])
    else:
        return np.array(dataset)

data, label = generate_dataset(TARGET_IMAGES, LABEL_FILEPATH)

train_set_x = data[:TRAIN_SET_COUNT]
val_set_x = data[TRAIN_SET_COUNT+1:TRAIN_SET_COUNT+VALIDATION_SET_COUNT]
test_set_x = data[TRAIN_SET_COUNT+VALIDATION_SET_COUNT+1:TRAIN_SET_COUNT+VALIDATION_SET_COUNT+TEST_SET]
train_set_y = label[:TRAIN_SET_COUNT]
val_set_y = label[TRAIN_SET_COUNT+1:TRAIN_SET_COUNT+VALIDATION_SET_COUNT]
test_set_y = label[TRAIN_SET_COUNT+VALIDATION_SET_COUNT+1:TRAIN_SET_COUNT+VALIDATION_SET_COUNT+TEST_SET]

train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, val_set_y

dataset = [train_set, val_set, test_set]

f = gzip.open(PKL_FILE, 'wb')
pickle.dump(dataset, f, protocol=2)
f.close()