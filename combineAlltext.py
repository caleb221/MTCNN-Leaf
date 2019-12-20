# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 21:47:21 2019

@author: C.Seifert18
"""

'''
combine 3 files...fuck this  computer up...
'''

import os
import pdb


outpath='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\MTCNN-Tensorflow-master\\Ara2013-Canon\\'
araPath='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\MTCNN-Tensorflow-master\\Ara2013-Canon\\ara2013_TFmtcnn_bb.txt'
leaf100Path='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\100-leaves-plant-species\\100-leaves-plant-species\\100species_bbx.txt'
plantVillagePath='C:\\Users\\c.seifert18\\Desktop\\trainMTCNN_stuff\\100-leaves-plant-species\\100-leaves-plant-species\\plantVillageTrain_bbx.txt'
Allplants=outpath+'allPlants_bb.txt'



with open(Allplants,'w') as newFile:
    with open (araPath,'r') as araFiles:    
        allText = araFiles.readlines()      
        for i in allText:
            i=i.strip().split(' ')       
            ara_win = outpath+'train\\'+i[0][154:]+" "#[p.strip().split(' ')[0] for p in allText if 'C:' not in p]
           
            newFile.writelines(ara_win)
            #pdb.set_trace()
            for k in i[1:]:
                newFile.writelines(k+" ")            
            newFile.writelines("\n")
        
        newFile.writelines("\n")
        
    #newFile.writelines("\n")    
    with open(leaf100Path, 'r') as binLeaf:
        all100  = binLeaf.readlines()
        
        for line in all100:            
#            if '\n' or '' in line:
#                 newFile.writelines('\n') 
#                 continue
            
            line = line.strip().split()
            thisDir = line[0]
            newFile.writelines(thisDir+' ')
            print(thisDir)
            for k in line[1:]:
                
                newFile.writelines(k+" ") 
            newFile.writelines("\n")
#    
#    with open(plantVillagePath, 'r') as villageLeaf:
#        allvillage  = villageLeaf.readlines()
#        for i in allvillage:
#            i = i.strip().split()
#            dirr=i[0]
#            newFile.writelines(dirr+" ")
#            for k in i[1:]:
#                newFile.writelines(k+' ')
#            newFile.writelines("\n")
        
        #village = [p.strip().split(' ')[0] for p in allvillage]
    binLeaf.close() 

binLeaf.close()
araFiles.close()
newFile.close()     