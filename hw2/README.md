# Instructions for hw2

In this homework we will perform a video classification task using visual features. The document/code is extended from the baseline repo provided by TAs.


## Overview
A video-based MED system is composed of mainly three parts: 1) Video pre-processing, 2) video feature extraction, and 3) classification. 


## Extract SURF features
For the hand-crafted visual feature, we ask you to Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ..., etc.). We use OpenCV to extract SURF feature. To do so, you need to downgrade OpenCV:
```
$ sudo pip uninstall opencv-python -y
$ sudo pip uninstall opencv-contrib-python -y
$ sudo pip install opencv-contrib-python==3.4.2.16
```

+ Step 1. Extract SURF keypoint features 
```
$ python surf_feat_extraction.py ./videos/ ./feature/
```

+ Step 2. K-Means clustering

First randomly select a subset of the features to speed things up. This may affect the quality of the clusters.
```
$ python select_frames.py labels/trainval.csv surf_feat/ 0.01 selected.surf.csv
```

K-Means clustering with 50 centers:
```
$ python train_kmeans.py selected.surf.csv 50 surf.kmeans.50.model
```

+ Step 3. Get Bag-of-Words features
```
$ ls videos/|while read line;do filename=$(basename $line .mp4);echo $filename;done > videos.name.lst
$ python get_bof.py surf.kmeans.50.model surf_feat/ 50  videos.name.lst surf_bof
```


## Extract CNN features
I used the ResNeXT-101 pretrained on Kinetics dataset for feature extraction. The code base I used is [Here](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).
```
$ python main.py  --input /labels/trainval.csv --video_root /videos --output /output/ --model resnext-101-64f-kinetics.pth --model_name resnext --model_depth 101 --mode feature --resnet_shortcut B --batch_size 32
```

## Training and Testing Classifiers
Train MLP model with SURF features:
```
$ python train_bof.py /trainval.csv /surf_bof/
```
Predict MLP model with SURF features:
```
$ python predict_bof.py /test_for_student.label /surf_bof/
```
Train MLP model with CNN features:
```
$ python train_tf.py /trainval.csv /out/
```
Predict MLP model with CNN features:
```
$ python predict_tf.py /test_for_student.label /out/
```