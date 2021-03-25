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
output_file = 'models/best.model'

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


n = len(label_list)
inds = np.arange(n)
np.random.shuffle(inds)

label_list = np.array(label_list)
feat_list = np.array(feat_list)
print('feat_list shape ',feat_list.shape)

print('features stats', np.mean(feat_list), np.std(feat_list))



y = np.array(label_list)
X = np.array(feat_list)

clf = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", \
    max_iter=1000, early_stopping=True, alpha=5e-4, n_iter_no_change=4)
clf.fit(X, y)

# save trained SVM in output_file
pickle.dump(clf, open(output_file, 'wb'))
print('Elapsed Time ', time.time() - start_time, ' seconds')
print('training accuracy: ', accuracy_score(y, clf.predict(X)))



# Now test the model


start_time = time.time()
# 1. load mlp model
mlp = pickle.load(open(output_file, "rb"))

# 2. Create array containing features of each sample
fread = open('labels/test_for_student.label', "r")
feat_list = []
video_ids = []
for line in fread.readlines():
  # HW00006228
  video_id = os.path.splitext(line.strip())[0]
  video_ids.append(video_id)

  feat = np.array([])

  # Add video features
  for vid_feat_dir in vid_feat_dirs:
    m, s = means[vid_feat_dir], stds[vid_feat_dir]
    feat_filepath = os.path.join(vid_feat_dir, video_id + '.csv')
    if not os.path.exists(feat_filepath):
      feat = np.concatenate((feat, (np.zeros(256) - m) / s))
    else:
      feat = np.concatenate((feat, (np.loadtxt(feat_filepath) - m) / s))

  # Add soundnet features
  for layer in soundnet_layers:
    m, s = means['soundnet_' + str(layer)], stds['soundnet_' + str(layer)]
    feat_filepath = os.path.join(soundnet_dir, video_id + 'tf_fea{}.npy'.format(str(layer).zfill(2)))
    if os.path.exists(feat_filepath):
      sn_feat = np.load(feat_filepath)
      layer_feat = np.concatenate((np.min(sn_feat, axis=0), \
        np.max(sn_feat, axis=0), np.mean(sn_feat, axis=0),\
        np.std(sn_feat, axis=0), np.quantile(sn_feat, 0.25, axis=0), np.quantile(sn_feat, 0.75, axis=0)))
      feat = np.concatenate((feat, (layer_feat-m)/s))
      layer_shapes[layer] = layer_feat.shape
    else:
      feat = np.concatenate((feat, np.zeros(layer_shapes[layer])))


  feat_list.append(feat)

X = np.array(feat_list)

start_time = time.time()
# 3. Get predictions
# (num_samples) with integer
pred_classes = mlp.predict(X)

# 4. save for submission
with open('early_fusion_best.csv', "w") as f:
  f.writelines("Id,Category\n")
  for i, pred_class in enumerate(pred_classes):
    f.writelines("%s,%d\n" % (video_ids[i], pred_class))

print('test run time', time.time() - start_time, 'seconds')