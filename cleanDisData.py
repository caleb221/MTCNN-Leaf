#
# 
#ok get the csv files to text
# do it for all (use blob)
#FAAACCCKKKKKKK
import csv
import os

def nameG(a, b): #get the file's names which are going to deal with
    c = []
    for i in range(b):
        if i==0:
            continue
        if i < 10:
            c.append(a +"tobacco_plant00"+ str(i)+"_bbox.csv")
        if i > 10 and i <100:
            c.append(a+"tobacco_plant0"+str(i)+"_bbox.csv")
        if i > 100:
            c.append(a+"tobacco_plant"+str(i)+"_bbox.csv")

    return c


dataPath=Path='/home/csse/DATASETS/DataSets/mtcnn-head-detection-master'
#dataPath = "/home/caleb/Downloads/DataSets/plantDB/Phenotyping_Leaf_detection_dataset/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Tobacco/"
outPath=   "/home/caleb/Downloads/Datasets/plantDB/Phenotyping_Leaf_detection_dataset/"
#csv_file = raw_input(dataPath +"ara2012_plant001_bbox.csv")
txt_file = "/home/caleb/Downloads/DataSets/plantDB/Phenotyping_Leaf_detection_dataset/bbox_all_tobacco.txt"


fileNames = nameG(dataPath,121)
print(fileNames)

with open(txt_file, "w") as txtOut:
    for i in fileNames: 
        with open(i,"r") as csvIn:
            txtOut.write(i[0:len(i)-8]+"rgb.png")
            txtOut.write(" ")
            [txtOut.write(" ".join(row)+"") for row in csv.reader(csvIn)]
            txtOut.write("\n")
txtOut.close()

