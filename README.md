# 11-775 Homework 3

Peter Schaldenbrand <br/>
11-775 Spring 2021<br/>
Homework 3 <br/>
Andrew ID: pschalde <br/>
GitHub ID: pschaldenbrand <br/>
Kaggle ID: pittsburghskeet

## Features

In this project I used both the video and audio features.  The video features were extracted from the torchvision video models: resnet18, r3d_18, mc3_18, and r2plus1d_18.  The features were extracted and averaged at the avgpool layer.  The audio features were extracted from the soundnet model at the 18th and 21st layers.  I took the mean, min, max, quartiles, and standard deviation of the features across the variable length dimension to create the audio features.

## Early Fusion

### Normalizing data

Run this script to cached the training data in single files per feature and compute the statistics for the features.

```python feature_stats.py```
```python cache_test_data.py```

From this script we can now figure out how to normalize our features to a normal distribution (by subtracting the mean and dividing by the standard deviation).  It's important to save these means since they are only calculated on the training and validation data and we need to apply them to the test data too.

Now we test schemes for normalization in the early fusion test.  In this test, we use 5-fold cross validation and average the validation accuracy accross folds.  If we do no normalization we get a validation accuracy of 96.75%.   If we normalize each feature before combining them, we get a 97.35% accuracy. If we normalize all the features after combining, we get a 97.23% accuracy.  And if we normalize before and after combination, we get 96.84% accuracy.  It appears that normalization doesn't make a huge difference since the model is so accurate already.  Nonetheless we will stick with normalizing the features prior to combining them.

These accuracies are far better than just audio alone which i was able to get to 66.9% validation accuracy.  The accuracies are as good as just using video since video got 97.2% validation accuracy.

### The model

I used a single layer 100 hidden unit MLP for the model.  It was trained with early stopping which was determined using 10% validation data.  This model got a 96.842% on the kaggle leaderboard.

To train: 

```python train_best_early_fusion.py```

## Late Fusion

My late fusion model use MLP's with one hidden layer with 100 hidden units for the models.  A model was trained on each set of features from video and audio.  Each model produced a size 10 vector of probabilities for the classes.  These probabilities were concatenated then another model was trained on them.  This resulted in a 98.25% validation accuracy when testing with 5-fold cross validation.  This was the best validation accuracy I have come accross so far.


These accuracies are far better than just audio alone which i was able to get to 66.9% validation accuracy.  The accuracies are better than video alone too since video got 97.2% validation accuracy.

This model took 41.24 seconds to train.

So I considered this my best model and it scored my personal record 97.368% on the Kaggle test data. Late fusion does have an improvement over early fusion.

To train:
```python train_best_late_fusion.py```

Here is the confusion matrix on the validation folds of the 5-fold cross validation test, where the row indicates the true value and the column indicates the predicted value.

|                             | dribbling <br>basketball | mowing <br>lawn | playing <br>guitar | playing <br>piano | playing <br>drums | tapping<br> pen | blowing <br>out <br>candles | singing | tickling | shoveling <br>snow |
|-----------------------------|--------------------------|-----------------|--------------------|-------------------|-------------------|-----------------|-----------------------------|---------|----------|--------------------|
| dribbling <br>basketball    | 597                      | 1               | 1                  | 0                 | 1                 | 1               | 0                           | 0       | 0        | 0                  |
| mowing <br>lawn             | 0                        | 600             | 0                  | 0                 | 0                 | 0               | 0                           | 0       | 0        | 1                  |
| playing <br>guitar          | 0                        | 1               | 582                | 1                 | 5                 | 0               | 2                           | 9       | 1        | 0                  |
| playing <br>piano           | 0                        | 0               | 1                  | 501               | 1                 | 0               | 0                           | 6       | 0        | 0                  |
| playing <br>drums           | 0                        | 0               | 3                  | 2                 | 585               | 1               | 1                           | 9       | 0        | 0                  |
| tapping<br> pen             | 0                        | 0               | 1                  | 0                 | 0                 | 521             | 1                           | 2       | 1        | 0                  |
| blowing <br>out <br>candles | 0                        | 0               | 1                  | 0                 | 1                 | 0               | 593                         | 4       | 2        | 0                  |
| singing                     | 1                        | 0               | 10                 | 2                 | 5                 | 0               | 6                           | 573     | 2        | 2                  |
| tickling                    | 0                        | 1               | 3                  | 0                 | 0                 | 0               | 3                           | 3       | 410      | 0                  |
| shoveling <br>snow          | 0                        | 0               | 0                  | 0                 | 0                 | 0               | 0                           | 0       | 0        | 601                |

THis model is exceptionally accurate, however it does make some notable mistakes.  For instance singing is often predected to be guitar and vice versa.  In general, singing is a hard category to predict correctly.  This is likely because singing correlates well visually and sonically with other instruments like guitar.  The model perfectly correctly classified all instances of shoveling snow.