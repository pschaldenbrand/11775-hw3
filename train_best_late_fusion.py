#!/bin/python

import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import pickle
import argparse
import sys
import pdb
import time


list_videos = 'labels/trainval.csv'
vid_feat_dirs = ['resnet_avgpool_feat', 'resnet_3d_feat', 'r2plus1d_18_feat', 'mc3_18_feat']
soundnet_dir = 'SoundNet-tensorflow/output'
soundnet_layers = [18, 21]

np.random.seed(0)

start_time = time.time()

# 1. read all features in one array.
fread = open(list_videos, "r")
feat_list = []
# labels are [0-9]
label_list = []
# load video names and events in dict
df_videos_label = {}
for line in open(list_videos).readlines()[1:]:
  video_id, category = line.strip().split(",")
  df_videos_label[video_id] = category

feat_shape = {}
layer_shapes = {}

for line in fread.readlines()[1:]:
  video_id = line.strip().split(",")[0]
  feat = np.array([])

  label_list.append(int(df_videos_label[video_id]))


feat_list = []
for feat_name in vid_feat_dirs + ['soundnet_18', 'soundnet_21']:
  with open('cache/' + feat_name + '.npy', 'rb') as f:
    feat_list.append(np.load(f))


n = len(label_list)
inds = np.arange(n)
np.random.shuffle(inds)


y = np.array(label_list)

models = []
late_fusion_train_x = None
for i in range(len(feat_list)):

  train_feat_list = feat_list[i]

  X = np.array(train_feat_list)

  clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
      max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=4)
  clf.fit(X, y)
  models.append(clf)

  late_fusion_train_x = clf.predict_proba(X) if late_fusion_train_x is None else np.concatenate((late_fusion_train_x, clf.predict_proba(X)), axis=1)


late_fusion_model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
      max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=4)
late_fusion_model.fit(late_fusion_train_x, y)


# save trained SVM in output_file
# pickle.dump(clf, open(output_file, 'wb'))
print('Elapsed Time ', time.time() - start_time, ' seconds')
print('training accuracy: ', accuracy_score(y, late_fusion_model.predict(late_fusion_train_x)))



# Now test the model


start_time = time.time()

# 2. Create array containing features of each sample
fread = open('labels/test_for_student.label', "r")
feat_list = []
video_ids = []
for line in fread.readlines():
  # HW00006228
  video_id = os.path.splitext(line.strip())[0]
  video_ids.append(video_id)

feat_list = []
for feat_name in vid_feat_dirs + ['soundnet_18', 'soundnet_21']:
  with open('cache_test/' + feat_name + '.npy', 'rb') as f:
    feat_list.append(np.load(f))

X_late_fusion = None

for i in range(len(feat_list)):
  X = np.array(feat_list[i])
  X_late_fusion = models[i].predict_proba(X) if X_late_fusion is None else np.concatenate((X_late_fusion, models[i].predict_proba(X)), axis=1)

start_time = time.time()
# 3. Get predictions
# (num_samples) with integer
pred_classes = late_fusion_model.predict(X_late_fusion)

# 4. save for submission
with open('late_fusion_best.csv', "w") as f:
  f.writelines("Id,Category\n")
  for i, pred_class in enumerate(pred_classes):
    f.writelines("%s,%d\n" % (video_ids[i], pred_class))

print('test run time', time.time() - start_time, 'seconds')