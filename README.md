MTCNN head detection
==================================

## Set up

Set up environment and copy C++ layer code to Caffe's source code tree.

```
$ export PYTHONPATH=/path/to/mtcnn-head-detection:$PYTHONPATH
$ export CAFFE_HOME=/path/to/caffe
$ pip install easydict
$ pip install lmdb
$ sh layers/copy.sh
```

Compile Caffe following its document.

## Prepare data

Download dataset [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release).
Unzip and put them in data directory.

## Train

**pnet**
```
python jfda/prepare.py --net p --wider --worker 8
python jfda/train.py --net p --gpu 0 --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25
```
**rnet**

Choose appropriate pnet caffemodel to generate prior for rnet, and edit ```cfg.PROPOSAL_NETS``` in ```config.py```
```
python jfda/prepare.py --net r --gpu 0 --detect --wider --worker 4
python jfda/train.py --net r --gpu 0 --size 128 --lr 0.05 --lrw 0.1 --lrp 5 --wd 0.0001 --epoch 25
```
**onet**

Choose appropriate rnet caffemodel to generate prior for onet, and edit ```cfg.PROPOSAL_NETS``` in ```config.py```
```
python jfda/prepare.py --net o --gpu 0 --detect --wider --worker 4
python jfda/train.py --net o --gpu $GPU --size 64 --lr 0.05 --lrw 0.1 --lrp 7 --wd 0.0001 --epoch 35
```

## Test

```
python simpledemo.py
```

## Note

1. Landmark alignment in original mtcnn is removed in this repo. Here only do object classification and bounding box regression. 

2. Each convolutional layer kernel number in onet has reduced for faster network inference.

## Result

**pnet**

![pnet1](https://user-images.githubusercontent.com/16308037/53081537-059a1180-3536-11e9-8aa6-4ecfa8639bee.jpg)

**rnet**

![rnet1](https://user-images.githubusercontent.com/16308037/53081792-7fca9600-3536-11e9-8341-16b176bb9b12.jpg)

**onet**

![onet1](https://user-images.githubusercontent.com/16308037/53081747-6c1f2f80-3536-11e9-84bc-6885cf991468.jpg)

## References

- [A Convolutional Neural Network Cascade for Face Detection](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Li_A_Convolutional_Neural_2015_CVPR_paper.pdf)
- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](http://arxiv.org/abs/1604.02878)
- [MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- [opencv-mtcnn](https://github.com/ksachdeva/opencv-mtcnn)
