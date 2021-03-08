#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
import pickle
import argparse
import sys
import pdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Train SVM


"""
python train_svm_multiclass.py ./bof/ ./trainval.csv
python train_svm_multiclass.py ./sound_out/ ./trainval.csv --feat_appendix .npy
"""

# parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
# parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
# parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

"""
"""
if __name__ == '__main__':
  args = parser.parse_args()

  # 1. read all features in one array.
  fread = open(args.list_videos, "r")
  feat_list = []
  # labels are [0-9]
  label_list = []
  # load video names and events in dict
  df_videos_label = {}
  for line in open(args.list_videos).readlines()[1:]:
    video_id, category = line.strip().split(",")
    df_videos_label[video_id] = category


  for line in fread.readlines()[1:]:
    video_id = line.strip().split(",")[0]
    feat_filepath = os.path.join(args.feat_dir, video_id + args.feat_appendix)
    # for videos with no audio, ignore
    if os.path.exists(feat_filepath):
      if args.feat_appendix == '.npy':
        feat = np.load(feat_filepath)
        feat.resize((8, 8))
        feat_list.append(feat.flatten())
      else:
        feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))

      label_list.append(int(df_videos_label[video_id]))

  print("number of samples: %s" % len(feat_list))
  y = np.array(label_list)
  X = np.array(feat_list)
  X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,
                                                      stratify=y,
                                                      random_state=1)
  # pass array for svm training
  # one-versus-rest multiclass strategy
  clf = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf")
  clf.fit(X_train, Y_train)

  evaluation = clf.score(X_test, Y_test)
  pred = clf.predict(X_test)
  matrix = confusion_matrix(Y_test, pred)

  print('evaluation = ', evaluation)

  print(matrix)
