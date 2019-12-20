#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:08:13 2019

@author: caleb
"""

import caffe
import numpy as np
import argparse
import os

def extract_caffe_model(model, weights, output_path):
  """extract caffe model's parameters to numpy array, and write them to files
  Args:
    model: path of '.prototxt'
    weights: path of '.caffemodel'
    output_path: output path of numpy params 
  Returns:
    None
  """
  net = caffe.Net(model, caffe.TEST)
  net.copy_from(weights)

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  for item in net.params.items():
    name, layer = item
    print('convert layer: ' + name)

    num = 0
    for p in net.params[name]:
      np.save(output_path + '/' + str(name) + '_' + str(num), p.data)
      num += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="model prototxt path .prototxt")
  parser.add_argument("--weights", help="caffe model weights path .caffemodel")
  parser.add_argument("--output", help="output path")
  args = parser.parse_args()
  extract_caffe_model(args.model, args.weights, args.output)

'''
#import sys
#sys.path.insert(0, 'python/')
import caffe
from caffe.proto import caffe_pb2

net_param = caffe_pb2.NetParameter()
with open('p_solver_iter_9900.caffemodel', 'r') as f:
    net_str = f.readlines()
net_param.ParseFromString(net_str)

print( net_param.layer[0].name)  # first layer
print (net_param.layer[-1].name ) # last layer


from caffe.proto import caffe_pb2
import google.protobuf.text_format
net = caffe_pb2.NetParameter()
f = open('model.prototxt', 'r')
net = google.protobuf.text_format.Merge(str(f.read()), net)
f.close()
for i in range(0, len(net.layer)):
    if net.layer[i].type == 'Convolution':
        if net.layer[i].convolution_param.bias_term == True:
            print 'layer has bias'
'''