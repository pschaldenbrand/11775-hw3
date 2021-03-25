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

means = {'resnet_avgpool_feat' : 0.9893, 
         'resnet_3d_feat' : 0.5858, 
         'r2plus1d_18_feat' : 0.68045, 
         'mc3_18_feat' : 0.74313,
         'soundnet_18' : 0.3674,
         'soundnet_21' : 0.28715}
stds = {'resnet_avgpool_feat' : 0.86498, 
         'resnet_3d_feat' : 0.51375, 
         'r2plus1d_18_feat' : 0.52052, 
         'mc3_18_feat' : 0.53418,
         'soundnet_18' : 0.71402,
         'soundnet_21' : 1.09565}

normalize = True

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

  # # Add video features
  # for vid_feat_dir in vid_feat_dirs:
  #   feat_filepath = os.path.join(vid_feat_dir, video_id + '.csv')
  #   if not os.path.exists(feat_filepath):
  #     feat = np.concatenate((feat, np.zeros(256)))
  #   else:
  #     feat = np.concatenate((feat, np.loadtxt(feat_filepath)))

  # # Add soundnet features
  # for layer in soundnet_layers:
  #   feat_filepath = os.path.join(soundnet_dir, video_id + 'tf_fea{}.npy'.format(str(layer).zfill(2)))
  #   if os.path.exists(feat_filepath):
  #     sn_feat = np.load(feat_filepath)
  #     layer_feat = np.concatenate((np.min(sn_feat, axis=0), \
  #       np.max(sn_feat, axis=0), np.mean(sn_feat, axis=0),\
  #       np.std(sn_feat, axis=0), np.quantile(sn_feat, 0.25, axis=0), np.quantile(sn_feat, 0.75, axis=0)))
  #     feat = np.concatenate((feat, layer_feat))
  #     layer_shapes[layer] = layer_feat.shape
  #   else:
  #     feat = np.concatenate((feat, np.zeros(layer_shapes[layer])))


  # feat_list.append(feat)

feat_list = None
for feat_name in vid_feat_dirs + ['soundnet_18', 'soundnet_21']:
  with open('cache/' + feat_name + '.npy', 'rb') as f:
    feat = np.load(f)
    if normalize: 
      feat = (feat - means[feat_name]) / stds[feat_name]
    if feat_list is None:
      feat_list = feat 
    else:
      feat_list = np.concatenate((feat_list, feat), axis=1)

# feat_list = (feat_list - np.mean(feat_list)) / np.std(feat_list)

n = len(label_list)
inds = np.arange(n)
np.random.shuffle(inds)

label_list = np.array(label_list)
feat_list = np.array(feat_list)
print('feat_list shape ',feat_list.shape)

print('features stats', np.mean(feat_list), np.std(feat_list))


for param in [1]:
  # print(param)

  conf_mat = None

  all_val_acc = []
  all_train_acc = []

  for fold in range(folds):
    start_val = int(n * (float(fold)/folds))
    end_val = min(int(n * (float(fold+1)/folds)), n)

    train_fold_inds = np.concatenate((inds[:start_val], inds[end_val:]))
    val_fold_inds = inds[start_val:end_val]

    train_label_list = label_list[train_fold_inds]
    train_feat_list = feat_list[train_fold_inds]

    val_label_list = label_list[val_fold_inds]
    val_feat_list = feat_list[val_fold_inds]

    y = np.array(train_label_list)
    X = np.array(train_feat_list)

    clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
        max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=4)
    clf.fit(X, y)

    y_val = np.array(val_label_list)
    X_val = np.array(val_feat_list)
    acc = accuracy_score(y_val, clf.predict(X_val))

    cf = confusion_matrix(y_val, clf.predict(X_val))
    conf_mat = cf if conf_mat is None else conf_mat + cf

    all_train_acc.append(accuracy_score(y, clf.predict(X)))
    all_val_acc.append(acc)

  # print(conf_mat)

  # save trained SVM in output_file
  print('Elapsed Time ', time.time() - start_time, ' seconds')
  print('Average training accuracy: ', np.array(all_train_acc).mean())
  print('Average validation accuracy: ', np.array(all_val_acc).mean())