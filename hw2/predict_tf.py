
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path

import argparse
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("test_path") # trainval.csv
# parser.add_argument("feat_dim", type=int)
parser.add_argument("feature_dir") # sound_out
# parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
args = parser.parse_args()

def get_data(test_path, feature_dir):
    test_feature_npy = []
    test_lines_final = []
    with open(test_path, 'r') as test_f:
        test_lines = test_f.readlines()
        random.shuffle(test_lines)
        for line in tqdm(test_lines):
            name = line.strip()
            file_path = Path(feature_dir+name.split('.')[0]+'.npy')
            if file_path.is_file():
                feature = np.load(feature_dir + name.split('.')[0] + '.npy')
                feature = np.mean(feature, axis=0)
                test_feature_npy.append(feature)
                test_lines_final.append(line)
    return np.asarray(test_feature_npy), np.asarray(test_lines_final)


test_path = args.test_path
feature_dir = args.feature_dir
test_examples, test_lines = get_data(test_path, feature_dir)

test_dataset = tf.data.Dataset.from_tensor_slices((test_examples))

BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 100

test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')

])


model.load_weights("./ckpt-0022.ckpt")
predictions  = model.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)

with open('test_resnext.csv', 'w') as out_f:
    for i in range(np.shape(predictions)[0]):
        print(test_lines[i].strip().split('.')[0]+','+str(predictions[i]), file=out_f)