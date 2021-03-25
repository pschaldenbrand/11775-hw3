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
folds = 5


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

label_list = np.array(label_list)


for param in [1]:

  conf_mat = None

  all_val_acc = []
  all_train_acc = []

  for fold in range(folds):
    start_val = int(n * (float(fold)/folds))
    end_val = min(int(n * (float(fold+1)/folds)), n)

    train_fold_inds = np.concatenate((inds[:start_val], inds[end_val:]))
    val_fold_inds = inds[start_val:end_val]

    train_label_list = label_list[train_fold_inds]
    val_label_list = label_list[val_fold_inds]

    y = np.array(train_label_list)

    models = []
    late_fusion_train_x = None
    for i in range(len(feat_list)):

      train_feat_list = feat_list[i][train_fold_inds]

      X = np.array(train_feat_list)

      clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
          max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=4)
      clf.fit(X, y)
      models.append(clf)

      late_fusion_train_x = clf.predict_proba(X) if late_fusion_train_x is None else np.concatenate((late_fusion_train_x, clf.predict_proba(X)), axis=1)


    #print('late_fusion_train_x', late_fusion_train_x.shape)
    late_fusion_model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
          max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=4)
    late_fusion_model.fit(late_fusion_train_x, y)

    y_val = np.array(val_label_list)
    late_fusion_train_x_val = None
    for i in range(len(feat_list)):
      val_feat_list = feat_list[i][val_fold_inds]
      X_val = np.array(val_feat_list)

      late_fusion_train_x_val = models[i].predict_proba(X_val) if late_fusion_train_x_val is None else np.concatenate((late_fusion_train_x_val, models[i].predict_proba(X_val)), axis=1)
    
    acc = accuracy_score(y_val, late_fusion_model.predict(late_fusion_train_x_val))

    cf = confusion_matrix(y_val, late_fusion_model.predict(late_fusion_train_x_val))
    conf_mat = cf if conf_mat is None else conf_mat + cf

    all_train_acc.append(accuracy_score(y, late_fusion_model.predict(late_fusion_train_x)))
    all_val_acc.append(acc)

  print(conf_mat)

  # save trained SVM in output_file
  print('Elapsed Time ', time.time() - start_time, ' seconds')
  print('Average training accuracy: ', np.array(all_train_acc).mean())
  print('Average validation accuracy: ', np.array(all_val_acc).mean())