#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:39:42 2019

@author: caleb
"""

#
# 
#ok get the csv files to text
# do it for all (use blob)

import cv2
import os
import pdb
import numpy as np

import matplotlib.pyplot as plt 


def nameG(a, b,plantType): #get the file's names which are going to deal with
    c = []
    for i in range(b):
        if i==0:
            continue
        if i < 10:
            c.append(a +plantType+"_plant00"+ str(i)+"_centers.png")
        if i > 10 and i <100:
            c.append(a+plantType+"_plant0"+str(i)+"_centers.png")
        if i > 100:
            c.append(a+plantType+"_plant"+str(i)+"_centers.png")

    return c





outpath='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\MTCNN-Tensorflow-master\\Ara2013-Canon\\'
araPath='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\MTCNN-Tensorflow-master\\Ara2013-Canon\\ara2013_TFmtcnn_bb.txt'
leaf100Path='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\100-leaves-plant-species\\100-leaves-plant-species\\100species_bbx.txt'
plantVillagePath='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\100-leaves-plant-species\\100-leaves-plant-species\\plantVillageTrain_bbx.txt'

serverdir='/home/csse/DATASETS/DataSets/mtcnn-head-detection-master'
#csv_file = raw_input(dataPath +"ara2012_plant001_bbox.csv")
txt_file = outpath+"allLandmark.txt"

#fileNames = nameG(dataPath,121,pType)
#print(fileNames)

inFile=outpath+'allPlants_bb.txt'

linux_dir='/home/caleb/Downloads/DataSets/plantDB/Phenotyping_Leaf_detection_dataset/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/train/'



with open(inFile,'r') as fIn:
        allText = fIn.readlines()
        fileNames=[p.strip().split(' ')[0] for p in allText if 'C:' in p]  #[0][:-7]+'label.png' for p in allText]
        #pdb.set_trace()
fIn.close()

with open(txt_file, 'w') as f:
    for file in fileNames:
        #print(file)
        f.writelines(linux_dir+file[90:]+" ")
        img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
        
#-- Step 1: Detect the keypoints using SURF Detector
        minHessian = 1000
        detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
        keypoints = detector.detect(img)
        #keypoints.=
#-- Draw keypoints
        #img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        #cv2.drawKeypoints(img, keypoints, img_keypoints)
        count = 0
        for k in keypoints:
            if count > 9:
                continue
            x=int(k.pt[0])
            y=int(k.pt[1])
            w=int(k.size)
            f.writelines(str(x)+" "+str(y)+" "+ str(w)+" "+str( w)+" ")
        f.writelines("\n")    
            
#-- Show detected (drawn) keypoints
        #cv2.imshow('SURF Keypoints', img_keypoints)
        #cv2.waitKey(0)
        #pdb.set_trace()
f.close()        
        
        
        
        
        
        
        
        
        
        
        
#        imGray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#
#        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
## define range of blue color in HSV
#        lower_blue = np.array([0,0,0])
#        upper_blue = np.array([10,10,10])
## Threshold the HSV image to get only blue colors
#        mask = cv2.inRange(hsv, lower_blue, upper_blue)
#        blue_only = cv2.bitwise_and(img,img, mask= mask)
#        im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        f.writelines(file+" ")
#        for i,cnt in enumerate(contours):
#            cv2.drawContours(blue_only, cnt, -1, (0,0,255), 1)
#            mask = np.zeros(imGray.shape,np.uint8)
#            cv2.drawContours(mask,[cnt],0,255,-1)
#            pixelpoints = np.transpose(np.nonzero(mask))
#            
#            for j in range(0,11):#len(0,len(cnt))
#                f.writelines(str(cnt[j][0][0])+" "+str(cnt[j][0][1]))
#            """
#            f.writelines(" "+str(cnt[0][0][0])+" "+str(cnt[0][0][1]))
#            f.writelines(" "+str(cnt[len(cnt)-1][0][0])+" "+str(cnt[len(cnt)-1][0][1]))
#            """
#        f.writelines("\n")
##          f.writelines(""+cnt[0][0][0])
##          f.writelines("contour " + str(i) +" :" + str(cnt))  
#f.close()
#
#plt.imshow(mask)
"""
for i in range(0,len(ind[0])):
    cv2.rectangle(imGray,(ind[0][i],ind[1][i]), (ind[0][i],ind[1][i]) , (0,255,0) ,1)
plt.subplot(2,2,1)
plt.imshow(img)
plt.subplot(2,2,2)
plt.imshow(bwIm)

"" "
for i in range(0,img.shape[0]):
    for j in range(0,img.shape[1]):
        pixel = img.item(i, j)
        print pixel


with open(txt_file, "w") as txtOut:
    for i in fileNames: 
        with open(i,"r") as csvIn:
            txtOut.write(i[0:len(i)-8]+"rgb.png")
            txtOut.write(" ")
            [txtOut.write(" ".join(row)+"") for row in csv.reader(csvIn)]
            txtOut.write("\n")
txtOut.close()
"""
