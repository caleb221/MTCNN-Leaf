# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long
import os
import time
import logging
import cv2
import numpy as np
from jfda.config import cfg
import xml.etree.ElementTree as ET
import pdb
#import psutil
longPath=Path='/home/csse/DATASETS/DataSets/mtcnn-head-detection-master/'
#longPath="/home/caleb/Downloads/DataSets/plantDB/Phenotyping_Leaf_detection_dataset/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/"

def load_scutbrainwashcheat():
 #   longPath="/home/caleb/Downloads/DataSets/plantDB/Phenotyping_Leaf_detection_dataset/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/"
    train_image_dirs = [longPath+'Ara2013-Canon/train']
    train_data = []
    for trainimdir in train_image_dirs:
        print( 'parsing ' + trainimdir + ' ...')
        for name in os.listdir(trainimdir):
            
            if name[-4:] == '.png':
               # pdb.set_trace()
                impath = trainimdir + name
                labelpath = impath.replace('.jpg', '.xml')
                labelpath.replace('JPEGImages', 'Annotations')
                tree = ET.parse(labelpath)
                root = tree.getroot()
                bboxes = []
                for obj in root.iter('object'):
                    xmlbox = obj.find('bndbox')
                    xmin = int(xmlbox.find('xmin').text)
                    ymin = int(xmlbox.find('ymin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymax = int(xmlbox.find('ymax').text)
                    w = xmax - xmin
                    h = ymax - ymin
                    size = min(w, h)
                    # only large enough
                    if size > 12:
                        bbox = [xmin, ymin, xmax, ymax]
                        bboxes.append(bbox)
                if len(bboxes) > 0:
                    bboxes = np.asarray(bboxes, dtype=np.float32)
                    train_data.append([impath, bboxes])
    return (train_data, train_data)

def load_cheat():

    train_image_dirs = [longPath+'Ara2013-Canon/train/']
    train_data = []
    for trainimdir in train_image_dirs:
        print ('parsing ' + trainimdir + ' ...')
        for name in os.listdir(trainimdir):
            if name[-4:] == '.jpg':
                impath = trainimdir + name
                labelpath = impath.replace('.jpg', '.xml')
                labelpath = labelpath.replace('JPEGImages', 'Annotations')
                tree = ET.parse(labelpath)
                root = tree.getroot()
                bboxes = []
                for obj in root.iter('object'):
                    xmlbox = obj.find('bndbox')
                    xmin = int(xmlbox.find('xmin').text)
                    ymin = int(xmlbox.find('ymin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymax = int(xmlbox.find('ymax').text)
                    w = xmax - xmin
                    h = ymax - ymin
                    size = min(w, h)
                    # only large enough
                    if size > 12:
                        bbox = [xmin, ymin, xmax, ymax]
                        bboxes.append(bbox)
                if len(bboxes) > 0:
                    bboxes = np.asarray(bboxes, dtype=np.float32)
                    train_data.append([impath, bboxes])

    val_image_dirs = [longPath+'Ara2013-Canon/validation/']
    val_data = []
    for valimdir in val_image_dirs:
        print ('parsing ' + valimdir + ' ...')
        for name in os.listdir(valimdir):
            if name[-4:] == '.jpg':
                impath = valimdir + name
                labelpath = impath.replace('.jpg', '.xml')
                labelpath = labelpath.replace('JPEGImages', 'Annotations')
                tree = ET.parse(labelpath)
                root = tree.getroot()
                bboxes = []
                for obj in root.iter('object'):
                    xmlbox = obj.find('bndbox')
                    xmin = int(xmlbox.find('xmin').text)
                    ymin = int(xmlbox.find('ymin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymax = int(xmlbox.find('ymax').text)
                    w = xmax - xmin
                    h = ymax - ymin
                    size = min(w, h)
                    # only large enough
                    if size > 12:
                        bbox = [xmin, ymin, xmax, ymax]
                        bboxes.append(bbox)
                if len(bboxes) > 0:
                    bboxes = np.asarray(bboxes, dtype=np.float32)
                    val_data.append([impath, bboxes])

    return (train_data, val_data)

def load_wider():
  """load wider face dataset
  data: [img_path, bboxes]+
  bboxes: [x1, y1, x2, y2]
  """

  def get_dirmapper(dirpath):
    """return dir mapper for wider face
    """
    mapper = {}
    #print(dirpath)
    for d in os.listdir(dirpath):
        
      dir_id = d.split('--')[0]
      
      mapper[dir_id] = os.path.join(dirpath, d)
    return mapper
  #pdb.set_trace()
  train_mapper = get_dirmapper(os.path.join(longPath+cfg.WIDER_DIR))
  #print(train_mapper)
  val_mapper = get_dirmapper(os.path.join(longPath+cfg.WIDER_DIR))

  def gen(text, mapper):
    fin = open(text, 'r')
    allText = fin.readlines()
    result = []
    count = 0
    for line in allText:
        
    #while True:
      #line = fin.readline()
      #print(line)
      #pdb.set_trace()
      if not line: break  # eof
      name = line.strip()
      
      if '/home' in line:
          dir_id = name.split(' ')
          #print(dir_id)
          img_path = dir_id[0]
           
      #if count >  2:
      #    pdb.set_trace()
      count +=1
      face_n = len(line)#name.strip(' ').split())#

      bboxes = []
      #pdb.set_trace()
      #print(line            )
      for i in range(0,face_n):
        
        #line = fin.readline()
        #pdb.set_trace()
        #line = fin.readline().strip()
        #print(line)
        components = line.strip(' ').split()#line.split(' ')
        #pdb.set_trace()
        #print("===================")
        #print(components)
        
        #if not components:
        #    continue
        
        try:
            if '/' in components[0]:
                #print("SKIP DIR")
                continue
        except:
           #print(components)
           #print("comething weird with components.... skipping")
           continue
       
        #print(components)
        
        """
            landmark=[x for x in range(0,len(components))]
        for i in range(1,len(components)):
            if components[i] is '[':
                continue
            if components[i] is ']':
                continue
            landmark[i]=components[i]
        """
#        print(components[i])
        #pdb.set_trace()
        x, y, w, h = [float(_) for _ in components if '/' not in _ if '[' not in _ if ']' not in _ ]
        #print("\n\n\n\n==================")
        #print(x, y, w, h)
        size = min(w, h)
        # only large enough
        if size > 12:
          bbox = [x, y, w, h]
          bboxes.append(bbox)
          #print("BBOX ADDED!")
      # # for debug
      
        #img = cv2.imread(img_path)
        #pdb.set_trace()
        #for bbox in bboxes:
          #x, y, w, h = bbox
          #cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 1)
          #cv2.imshow('img', img)
          #cv2.waitKey(0)

      if len(bboxes) > 0:
        bboxes = np.asarray(bboxes, dtype=np.float32)
        bboxes[:, 2] += bboxes[:, 0]
        bboxes[:, 3] += bboxes[:, 1]
        result.append([img_path, bboxes])
   
    
    fin.close()
    if len(result) == 0:
     	print("[[NOTHING RETURNED]]")
	#pdb.set_trace()
    
    return result
  #pdb.set_trace()
  txt_dir = os.path.join(longPath+cfg.WIDER_DIR)
  train_data = gen(os.path.join(txt_dir, 'server_allPlants_bb.txt'), train_mapper)
  val_data = gen(os.path.join(txt_dir, 'server_allPlants_bb.txt'), val_mapper)
  
  return (train_data, val_data)


def load_celeba():
  """load celeba dataset and crop the face bbox
  notice: the face bbox may out of the image range
  data: [img_path, bbox, landmark]
  bbox: [x1, y1, x2, y2]
  landmark: [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5], align to top left of the image
  """
  text = os.path.join(longPath+cfg.CelebA_DIR, 'server_allLandmark.txt')
  
  with open(text, 'r') as fileIn:
      fin = fileIn.readlines()
  
  
  #pdb.set_trace()
  
  n = int(len(fin))#.readline().strip()))-125
  #fin.readline()  # drop

  result = []
  
  
  for i in fin:
    #print(n)
    line = i#fin.readline().split()
    #pdb.set_trace()
    components = line.split()
    #pdb.set_trace()
    #print(cfg.CelebA_DIR+ components[0].strip('['))
    print("\n")
    img_path = os.path.join(components[0])
    #pdb.set_trace()
    print('PATH    ',img_path)
    if '/home' in components:
        continue
    landmark=[x for x in range(0,10)]
    for i in range(1,10):
        if components[i] is '[':
            continue
        if components[i] is ']':
            continue
        if components[i] is ' ':
            continue
        landmark[i]=components[i]
    
    #print(landmark)
    landmark = np.asarray(landmark, dtype=np.float32)
    #print(landmark)
    #pdb.set_trace()
    landmark = landmark.reshape((-1, 2)) # 5x2
    #pdb.set_trace()
    # crop face bbox
    x_max, y_max = landmark.max(0)
    x_min, y_min = landmark.min(0)
    w, h = x_max - x_min, y_max - y_min
    w = h = max(w, h)
    ratio = 0.5
    x_new = x_min - w*ratio
    y_new = y_min - h*ratio
    w_new = w*(1 + 2*ratio)
    h_new = h*(1 + 2*ratio)
    bbox = [x_new, y_new, x_new + w_new, y_new + h_new]
    bbox = [int(_) for _ in bbox]

    # # for debug
    # img = cv2.imread(img_path)
    # x, y, w, h = bbox
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
    # for j in range(5):
    #   cv2.circle(img, (int(landmark[j, 0]), int(landmark[j, 1])), 2, (0,255,0), -1)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # # normalize landmark
    # landmark[:, 0] = (landmark[:, 0] - bbox[0]) / w_new
    # landmark[:, 1] = (landmark[:, 1] - bbox[1]) / h_new

    landmark = landmark.reshape(-1)
    result.append([img_path, bbox, landmark])

  #fin.close()
  ratio = 0.8
  train_n = int(len(result) * ratio)
  train = result[:train_n]
  val = result[train_n:]
  return train, val


def get_logger(name=None):
  """return a logger
  """
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  sh = logging.StreamHandler()
  sh.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s')
  sh.setFormatter(formatter)
  logger.addHandler(sh)
  return logger


def crop_face(img, bbox, wrap=True):
  height, width = img.shape[:-1]
  x1, y1, x2, y2 = bbox
  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
  if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
   # print ('[WARN] ridiculous x1, y1, x2, y2')
    return None
  if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
    # out of boundary, still crop the face
    if not wrap:
      return None
    
    h, w = y2 - y1, x2 - x1
    #print("Making big array....")
    #if (psutil.virtual_memory().available - psutil.virtual_memory().used) <= -150000000:
        #print("NO MEMORY!..skippping")
        #return None
    #print("mem: ",(psutil.virtual_memory().available-psutil.virtual_memory().used))
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    vx1 = 0 if x1 < 0 else x1
    vy1 = 0 if y1 < 0 else y1
    vx2 = width if x2 > width else x2
    vy2 = height if y2 > height else y2
    sx = -x1 if x1 < 0 else 0
    sy = -y1 if y1 < 0 else 0
    vw = vx2 - vx1
    vh = vy2 - vy1
    patch[sy:sy+vh, sx:sx+vw] = img[vy1:vy2, vx1:vx2]
    return patch
  return img[y1:y2, x1:x2]


class Timer:

  def __init__(self):
    self.start_time = 0
    self.total_time = 0

  def tic(self):
    self.start_time = time.time()

  def toc(self):
    self.total_time = time.time() - self.start_time

  def elapsed(self):
    return self.total_time


if __name__ == '__main__':
  img = cv2.imread(longPath+'/Ara2012/ara2012_plant001_rgb.png')
  bbox = [-100, -200, 300, 400]
  patch = crop_face(img, bbox)
  #cv2.imshow('patch', patch)
  #cv2.waitKey(0)
  print("test is done, you cant see the picture though cause you on a terminal\n....imagine it bitch")
