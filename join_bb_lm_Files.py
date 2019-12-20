#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 23:42:06 2019

@author: caleb
"""

#combine landmark and bbox data... landmark first then bbox


import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 

import pdb

outPath='/home/csse/DATASETS/DataSets/mtcnn-head-detector-master/Ara-2013-Canon/' 
# "/home/caleb/Downloads/DataSets/plantDB/Phenotyping_Leaf_detection_dataset/"
#csv_file = raw_input(dataPath +"ara2012_plant001_bbox.csv")
txt_file = 'serverDir_All_bb.txt'

in1bb =outPath+"bbox_all_ara2012.txt"
in2bb =outPath+"bbox_all_ara2013.txt" 
in3bb =outPath+"bbox_all_tobacco.txt" 

in1lm=outPath+"ara2012Landmark.txt" 
in2lm=outPath+"ara2013Landmark.txt"
in3lm=outPath+"tobaccoLandmark.txt"


#print(fileNames)
process_bbfiles = [in1bb,in2bb,in3bb]
process_lmfiles = [in1lm,in2lm,in3lm]



with open(txt_file, 'w') as f:
    for i in range(0,len(process_bbfiles)):
        with open(process_bbfiles[i],'r') as bbFile:
            bbcontents = bbFile.read
            es()
        with open(process_lmfiles[i]) as lmFile:
            lmcontents = lmFile.readlines()
            #contents= contents.split(" ",1)
        print(len(bbcontents))
        
        max_2012 = 118
        max_2013 = 107
        max_tobacco=61
        goTo =0
        
        if i == 0:
            goTo = max_2012
        if i ==1:
            goTo=max_2013
        if i==2:
            goTo=max_tobacco
            
        for j in range(0,goTo):#,len(bbcontents)):
            bbcoord = bbcontents[j].strip().split(' ')      
            lmcoord = lmcontents[j].strip().split(' ')
            
            #print(coord[0])
            
            try: 
                int(bbcoord[0])
                print("bad first char, skip!")
                continue
            except:
                #print("cool..not a number")
                print(".."+bbcoord)
            
            #/home/caleb/Downloads/DataSets/plantDB/Phenotyping_Leaf_detection_dataset/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/train
            if i==0:
                filePath=bbcoord[0][:140]+"3-Canon/train/"+bbcoord[0][142:]
                #pdb.set_trace()
            if i == 1:    
                filePath=bbcoord[0][:148]+"train/"+bbcoord[0][148:]
            if i== 2:
                filePath = bbcoord[0][:134]+'Ara2013-Canon/train/'+bbcoord[0][142:]
            #filePath=bbcoord[0]
            # f.writelines("[")
            f.writelines(filePath+" ")#"] [ ")
            #n=0
            for k in range(2,len(bbcoord)):
                f.writelines(bbcoord[k]+" ")
            '''
                n+=1
                if n ==4:
                    f.writelines("\n")
                    n=0
            '''        
            f.writelines("\n")
            #f.writelines(" ] [ ")
            #for k in range(4,14):
            #    f.writelines(lmcoord[k]+" ")
            #f.writelines(" ]\n")
            #bbFile.read()
        bbFile.close()
        lmFile.close()            
            #bbFile.close()
        
f.close()
