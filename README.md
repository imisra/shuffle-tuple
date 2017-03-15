## Shuffle and Learn (Shuffle Tuple)
Created by [Ishan Misra](http://www.cs.cmu.edu/~imisra/)

Based on the ECCV 2016 Paper - "Shuffle and Learn: Unsupervised Learning using Temporal Order Verification" [link to paper](http://arxiv.org/abs/1603.08561).

This codebase contains the model and training data from our paper.

### Introduction

Our code base is a mix of Python and C++ and uses the [Caffe](https://github.com/BVLC/caffe) framework.
Design decisions and some code is derived from the [Fast-RCNN codebase](https://github.com/rbgirshick/fast-rcnn) by Ross Girshick.

### Citing

If you find our code useful in your research, please consider citing:
```
@inproceedings{misra2016unsupervised,
  title={{Shuffle and Learn: Unsupervised Learning using Temporal Order Verification}},
  author={Misra, Ishan and Zitnick, C. Lawrence and Hebert, Martial},
  booktitle={ECCV},
  year={2016}
}
```
### Benchmark Results
We summarize the results of finetuning our method here (details in the paper).

**Action Recognition**

| Dataset | Accuracy (split 1) | Accuracy (mean over splits)
:--- | :--- | :--- | :---
UCF101 | 50.9 | 50.2
HMDB51 | 19.8 | 18.1

Pascal Action Classification (VOC2012): Coming soon

**Pose estimation**
- FLIC: PCK (Mean, AUC) 84.7, 49.6
- MPII: PCKh@0.5 (Upper, Full, AUC): 87.7, 85.8, 47.6

**Object Detection**
- PASCAL VOC2007 test mAP of 42.4% using Fast RCNN.

We initialize conv1-5 using our unsupervised pre-training. We initialize fc6-8 randomly.
We then follow the procedure from Krahenbuhl et al., 2016 to rescale our network and finetune all layers using their hyperparameters.

**Surface Normal Prediction**
- NYUv2 (Coming soon)

### Contents
1. [Requirements: software](#requirements-software)
2. [Models and Training Data](#models-and-training-data)
3. [Usage](#usage)
4. [Utils](#utils)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers and OpenCV.

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  USE_OPENCV := 1
  ```

You can download a compatible fork of Caffe from [here](https://github.com/BVLC/caffe/tree/8e8d97d6206cac99eae3c16baaa2275a14e64ca7). Note that since our model requires Batch Normalization, you will need to have a fairly recent fork of caffe.

### Models and Training Data
1. Our model trained on tuples from UCF101 (train split 1, without using action labels) can be downloaded [here](http://goo.gl/tzHrVH).

2. The tuples used for training our model can be downloaded as a zipped text file [here](http://goo.gl/QjEDxw). Each line of the file `train01_image_keys.txt` defines a tuple of three frames. The corresponding file `train01_image_labs.txt` has a binary label indicating whether the tuple is in the correct or incorrect order.

3. Using the training tuples requires you to have the raw videos from the UCF101 dataset ([link to videos](http://crcv.ucf.edu/data/UCF101/UCF101.rar)).
 We extract frames from the videos and resize them such that the max dimension is 340 pixels.
You can use `ffmpeg` to extract the frames. Example command: `ffmpeg -i <video_name> -qscale 1 -f image2 <video_sub_name>/<video_sub_name>_%06d.jpg`, where `video_sub_name` is the name of the raw video without the file extension.

### Usage
1. Once you have downloaded and formatted the UCF101 videos, you can use the `networks/tuple_train.prototxt` file to train your network. The only complicated part in the network definition is the data layer, which reads a tuple and a label. The data layer source file is in the `python_layers` subdirectory. Make sure to add this to your `PYTHONPATH`.
2. Training for Action Recognition: We used the codebase from [here](https://github.com/yjxiong/temporal-segment-networks)
3. Training for Pose Estimation: We used the codebase from [here](https://github.com/mitmul/deeppose). Since this code does not use `caffe` for training a network, I have included a experimental data layer for `caffe` in `python_layers/pose_data_layer.py`

### Utils
 This repo also includes a bunch of utilities I used for training and debugging my models
- `python_layers/loss_tracking_layer`: This layer tracks loss of each individual data point and its class label. This is useful for debugging as one can see the loss per class across epochs. Thanks to Abhinav Shrivastava for discussions on this.
- `model_training_utils`: This is the wrapper code used to train the network if one wants to use the `loss_tracking` layer. These utilities not only track the loss, but also keep a log of various other statistics of the network - weights of the layers, norms of the weights, magnitude of change etc. For an example of how to use this check `networks/tuple_exp.py`. Thanks to Carl Doersch for discussions on this.
- `python_layers/multiple_image_multiple_label_data_layer`: This is a fairly generic data layer that can read multiple images and data. It is based off my [data layers repo](https://github.com/imisra/caffe-data-layers).
