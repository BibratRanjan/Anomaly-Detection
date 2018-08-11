# Anomaly-Detection
Anomaly detection is an unsupervised learning task where the goal is to identify abnormal patterns or motions in data that are by definition infrequent or rare events. . Furthermore, anomalies are rarely annotated and labeled data rarely available to train a deep convolutional network to separate normal class from the anomalous class. This is a fairly complex task since the class of normal points includes frequently occurring objects and regular foreground movements while the anomalous class include various types of rare events and unseen objects that could be summarized as a consistent class. Long streams of videos containing no anomalies are made available using which one is required to build a representation for a moving window over the video stream that estimates the normal behavior class while detecting anomalous movements and appearance, such as unusual objects in the scene.

Given a set of training samples containing no anomalies, the goal of anomaly detection is to design or learn a feature representation, that captures “normal” motion and spatial appearance patterns. Any deviations from this normal can be identified by measuring the approximation error either geometrically in a vector space or the posterior probability of a given model which fits training sample representation vectors or by modeling the conditional probability of future samples given their past values and measuring the prediction error of test samples by training a predictive model, thus accounting for temporal structure in videos.

## Approach
We divide the video clip into sets of 4 frames for training and objective is to predict the 4th frame after seeing the first 3 frames. For this we have used the encoder-decoder model. The encoder reads the first 3 frames and derives a feature vector and then passes it through the decoder model to predict the 4th frame. 
```
The loss function used is the l2 loss function.
```

## Prerequisites
Keras 2.2 with tensorflow background.

