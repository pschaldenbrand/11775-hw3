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

cache_dir = 'cache'
if not os.path.exists(cache_dir): os.mkdir(cache_dir)

np.random.seed(0)

start_time = time.time()



# Video Feature stats
for vid_feat_dir in vid_feat_dirs:
  print(vid_feat_dir)

  feat = []
  for line in open(list_videos).readlines()[1:]:
    video_id, _ = line.strip().split(",")
    feat_filepath = os.path.join(vid_feat_dir, video_id + '.csv')
    if not os.path.exists(feat_filepath):
      feat.append(np.zeros(256))
    else:
      feat.append(np.loadtxt(feat_filepath))

  feat = np.array(feat)
  print(feat.shape, np.mean(feat), np.std(feat))
  print()
  with open(os.path.join(cache_dir, vid_feat_dir + '.npy'), 'wb') as f:
    np.save(f, feat)

layer_shapes = {}

# Audio Feature stats
for layer in soundnet_layers:
  print('soundnet ', layer)

  feat = []
  for line in open(list_videos).readlines()[1:]:
    video_id, _ = line.strip().split(",")
    feat_filepath = os.path.join(soundnet_dir, video_id + 'tf_fea{}.npy'.format(str(layer).zfill(2)))
    if os.path.exists(feat_filepath):
      sn_feat = np.load(feat_filepath)
      layer_feat = np.concatenate((np.min(sn_feat, axis=0), \
        np.max(sn_feat, axis=0), np.mean(sn_feat, axis=0),\
        np.std(sn_feat, axis=0), np.quantile(sn_feat, 0.25, axis=0), np.quantile(sn_feat, 0.75, axis=0)))
      feat.append(layer_feat)
      layer_shapes[layer] = layer_feat.shape
    else:
      feat.append(np.zeros(layer_shapes[layer]))

  feat = np.array(feat)
  print(feat.shape, np.mean(feat), np.std(feat))
  print()
  with open(os.path.join(cache_dir, 'soundnet_' + str(layer) + '.npy'), 'wb') as f:
    np.save(f, feat)

print('time ', time.time() - start_time)