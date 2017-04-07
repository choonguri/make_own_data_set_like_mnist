import gzip
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

PKL_FILE = "dataset.pkl.gz"
IMAGE_WIDTH = 64

f = gzip.open(PKL_FILE, "rb")
train_set, val_set, test_set = pickle.load(f)

data = np.array(train_set[0][1])
data = np.reshape(data, (-1, IMAGE_WIDTH, 3))

print(train_set[1])

img = Image.fromarray(data.astype('uint8'), 'RGB')

plt.imshow(img)
plt.show()