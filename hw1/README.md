# Homework 1
Here are the steps for running Homework 1's code. Some of the code/steps are borrowed from the baseline repo.

## Data and Labels

Please download the videos from [this link](https://drive.google.com/file/d/1Oyzv7eC0QDrg0vX3AdSXYzdsFpIsdzT-/view?usp=sharing) or [this link](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/hw_11775_videos.zip). Then download the labels from [here](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/labels_v2.zip).

## Step-by-step baseline instructions

First, put the videos into `videos/` folder and the labels into `labels/` folder simply by:
```
$ unzip hw_11775_videos.zip
$ unzip labels_v2.zip
```

### MFCC-Bag-Of-Features

Let's create the folders we need first:
```
$ mkdir audio/ mfcc/ bof/
```

1. Dependencies: FFMPEG, OpenSMILE, Python: sklearn, pandas

Download OpenSMILE 2.3.0 from [here](https://he.audeering.com/download/opensmile-2-3-0-tar-gz/) and then extract in this directory (another [link](https://aladdin-eax.inf.cs.cmu.edu/shares/11775/homeworks/opensmile-2.3.0.tar.gz) for the package):
```
$ tar -zxvf opensmile-2.3.0.tar.gz
```

Install FFMPEG by:
```
$ sudo apt install ffmpeg
```

Install python dependencies by:
```
$ sudo pip2 install sklearn pandas tqdm
```

2. Get MFCCs

We will first extract wave files and then run OpenSMILE to get MFCCs into CSV files. We will directly run the binaries of OpenSMILE (no need to install):
```
$ for file in videos/*;do filename=$(basename $file .mp4); ffmpeg -y -i $file -ac 1 -f wav audio/${filename}.wav; ./opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract -C config/MFCC12_0_D_A.conf -I audio/${filename}.wav -O mfcc/${filename}.mfcc.csv;done
```
The above should take 1-2 hours. *We got 7939 wav files and mfcc files out of 7942 videos*.

Note that some audio/mfcc files might be missing. This is due to the fact that some videos have no audio, which is common in real-world scenario. We'll learn to deal with that.

3. K-Means clustering

As taught in the class, we will use K-Means to get feature codebook from the MFCCs. Since there are too many feature lines, we will randomly select a subset (20%) for K-Means clustering by:
```
$ python2 select_frames.py labels/trainval.csv 0.2 selected.mfcc.csv --mfcc_path mfcc/
```

Now we train it by (50 clusters, this would take about 7-15 minutes):
```
$ python2 train_kmeans.py selected.mfcc.csv 50 kmeans.50.model
```

4. Feature extraction

Now we have the codebook, we will get bag-of-features (a.k.a. bag-of-words) using the codebook and the MFCCs. First, we need to get video names:
```
$ ls videos/|while read line;do filename=$(basename $line .mp4);echo $filename;done > videos.name.lst
```


Now we extract the feature representations for each video:
```
$ python2 get_bof.py kmeans.50.model 50 videos.name.lst --mfcc_path mfcc/ --output_path bof/
```

Now you can follow [here](#svm-classifier) to train SVM classifiers or [MLP](#mlp-classifier) ones.

### SoundNet-Global-Pool

Just as the MFCC-Bag-Of-Feature, we could also use the [SoundNet](https://arxiv.org/pdf/1610.09001.pdf) model to extract a vector feature representation for each video. Since SoundNet is trained on a large dataset, this feature is usually better compared to MFCCs.

Please follow [this Github repo](https://github.com/eborboihuc/SoundNet-tensorflow) to extract audio features. Please read the paper and think about what layer(s) to use. If you save the feature representations in the same format as in the `bof/` folder, you can directly train SVM and MLP using the following instructions.

I use the pool5 layer to extract features.
```
$ python extract_feat.py -t './demo.txt' -o "./sound_out/" -m 18 -x 19 -s -p extract
```

### SVM classifier

From the previous sections, we have extracted two fixed-length vector feature representations for each video. We will use them separately to train classifiers.

Suppose you are under `hw1` directory. Train SVM by:
```
$ mkdir models/
$ python train_svm_multiclass.py ./bof/ ./trainval.csv
$ python train_svm_multiclass.py ./sound_out/ ./trainval.csv --feat_appendix .npy

```


### MLP classifier

Suppose you are under `hw1` directory. Train MLP by:
```
$ python train_mlp.py ./bof/ ./trainval.csv
$ python train_mlp.py ./sound_out/ ./trainval.csv --feat_appendix .npy
```

### 1D CNN classifier

Suppose you are under `hw1` directory. Train 1D CNN by:
```
$ python train_cnn.py ./trainval.csv ./sound_out/ 
$ python train_cnn_mfcc.py ./trainval.csv ./bof/ 
```

Test 1D CNN by:
```
$ python predict_cnn.py ./test_for_student.label ./sound_out/ 
$ python predict_cnn_mfcc.py ./test_for_student.label ./bof/ 
```