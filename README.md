# MTCNN-Leaf
An experiment to re-purpose MTCNN for other uses than facial detection
This repo is the source code for implementing MTCNN using Caffe.

# Introduction
This was the first half of my Thesis project.
The goal is to re-purpose the MTCNN Facial detection model 
and use it on an ESP-32 AI-Thinker WiFi Camera module.
--> That code is in another repository: <a href="https://github.com/caleb221/ESP32-Leaf"> ESP32-Leaf</a>

# MTCNN
https://arxiv.org/pdf/1604.02878.pdf 

# SETUP
You'll need caffe installed, and setup can be done in the same way as 

https://github.com/lincolnhard/mtcnn-head-detection

There are a few Data preprocessing python scripts I also made to
help clean out the datasets I used. They are included at the top level directory.

There is also a python script that extracts the weights from a Caffe model and forms them into a numpy array file. This is used to translate the weights from one framework (caffe) to another (ESP-Face) The translation code is included in the ESP32-Leaf repository because it is a specific translation for that framework.


# Output
--> could use some improvement, but overall it found some leaves 



![Input Test](https://github.com/caleb221/MTCNN-Leaf/blob/master/test.png)
![P-Net Output](https://github.com/caleb221/MTCNN-Leaf/blob/master/pnet.jpg)
![R-Net Output](https://github.com/caleb221/MTCNN-Leaf/blob/master/rnet.jpg)
![O-Net Output](https://github.com/caleb221/MTCNN-Leaf/blob/master/onet.jpg)


# Data Sets
There were 2 datasets used to train the model
Found here

Phenotype Data set

https://www.plant-phenotyping.org/datasets-overview

100 Plant leaves

https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set


# References
The code that this half of the project was based off is found here

https://github.com/lincolnhard/mtcnn-head-detection
  
Special thanks to Lincolnhard and the owners of the Data sets
 
M. Minervini, A. Fischbach, H.Scharr, and S.A. Tsaftaris. Finely-grained annotated datasets for image-based plant phenotyping. Pattern Recognition

Charles Mallah, James Cope, James Orwell. Plant Leaf Classification Using Probabilistic Integration of Shape, Texture and Margin Features. Signal Processing, Pattern Recognition and Applications, in press. 2013.
