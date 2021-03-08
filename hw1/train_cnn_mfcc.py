
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path

import argparse
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras import Model, optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import random
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument("trainval_path") # trainval.csv
# parser.add_argument("feat_dim", type=int)
parser.add_argument("feature_dir") # bof
# parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
args = parser.parse_args()
# tf.debugging.set_log_device_placement(True)

def get_data(trainval_path, feature_dir):
    train_feature_npy = []
    train_label_npy = []
    val_feature_npy = []
    val_label_npy = []
    with open(trainval_path, 'r') as trainval_f:
        trainval_lines = trainval_f.readlines()
        random.shuffle(trainval_lines)
        train_lines = trainval_lines[:int(len(trainval_lines)*0.8)]
        val_lines = trainval_lines[int(len(trainval_lines)*0.8):]
        for line in tqdm(train_lines):
            # print('line = ', line)
            name, label = line.split(',')[0], line.split(',')[1]
            file_path = Path(feature_dir+name+'.csv')
            if file_path.is_file():
                # feature = np.load(feature_dir + name + '.npy')
                # feature.resize((54, 256))
                feature = np.genfromtxt(feature_dir + name + '.csv',
                                        delimiter=";", dtype="float")
                feature = np.expand_dims(feature, -1)
                train_feature_npy.append(feature)
                train_label_npy.append(int(label))

        for line in tqdm(val_lines):
            name, label = line.split(',')[0], line.split(',')[1]
            file_path = Path(feature_dir + name + '.csv')
            if file_path.is_file():
                # feature = np.load(feature_dir + name + '.npy')
                # feature.resize((54, 256))
                feature = np.genfromtxt(feature_dir + name + '.csv', delimiter=";", dtype="float")
                feature = np.expand_dims(feature, -1)
                val_feature_npy.append(feature)
                val_label_npy.append(int(label))


    return np.asarray(train_feature_npy), np.asarray(train_label_npy),\
           np.asarray(val_feature_npy), np.asarray(val_label_npy)



trainval_path = args.trainval_path
feature_dir = args.feature_dir


train_examples, train_labels, val_examples, val_label = get_data(
    trainval_path, feature_dir)


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_examples, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices(
    (val_examples, val_label))

BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(
    BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv1D(64, 3, padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv1D(128, 3, padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv1D(256, 3, padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')

])

lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.01,
                                 patience=1, verbose=1, mode='max')
lr = 0.0005


model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
checkpoint_path = "./ckpt-{epoch:04d}.ckpt"

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

model.fit(train_dataset, validation_data=test_dataset, epochs=50, callbacks=[lr_scheduler,ckpt_callback])

predictions=model.predict(test_dataset)
predictions = np.argmax(predictions, axis=1)

accu = np.sum(predictions==val_label) / np.shape(val_label)[0]
matrix = confusion_matrix(val_label, predictions)
