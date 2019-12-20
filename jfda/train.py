#!/usr/bin/env python2.7
# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long

import shutil
import argparse
import multiprocessing
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


import pdb

import sys
sys.path.insert(1, '/home/csse/DATASETS/DataSets/mtcnn-head-detection-master/')#/MTCNN-Tensorflow-master')

from jfda.config import cfg
from jfda.minibatch import MiniBatcher


longPath="/home/csse/DATASETS/DataSets/mtcnn-head-detection-master/"
longerPath=longPath+'jfda/'
class Solver:

  def __init__(self, solver_prototxt, args):
    net_type = args.net
    self.net_type = net_type
    input_size = cfg.NET_INPUT_SIZE[net_type]
    db_names_train = [longerPath+'data/%snet_negative_train'%net_type,
                      longerPath+'data/%snet_positive_train'%net_type,
                      longerPath+'data/%snet_part_train'%net_type]#'data/%scnet_part_train'%net_type]
    db_names_test = [longerPath+'data/%snet_negative_val'%net_type,
                     longerPath+'data/%snet_positive_val'%net_type,
                     longerPath+'data/%snet_part_val'%net_type]
    base_size = args.size
    ns = [r*base_size for r in cfg.DATA_RATIO[net_type]]
    # batcher setup
    
    batcher_train = MiniBatcher(db_names_train, ns, net_type)
    batcher_test = MiniBatcher(db_names_test, ns, net_type)
    # data queue setup
    #pdb.set_trace()
    queue_train = multiprocessing.Queue(32)#32
    queue_test = multiprocessing.Queue(32)#32
    batcher_train.set_queue(queue_train)
    batcher_test.set_queue(queue_test)
    # solver parameter setup
    size_train = batcher_train.get_size()
    size_test = batcher_test.get_size()
    iter_train = sum([x/y for x, y in zip(size_train, ns)]) / len(ns)  # train epoch size
    iter_test = sum([x/y for x, y in zip(size_test, ns)]) / len(ns)  # test epoch size
    max_iter = args.epoch * iter_train
    self.final_model = longPath+'tmp/%snet_iter_%d.caffemodel'%(net_type, max_iter)
    solver_param = caffe_pb2.SolverParameter()
    
    #pdb.set_trace()
    with open(longPath+solver_prototxt, 'r') as fin:
      text_format.Merge(fin.read(), solver_param)
    solver_param.max_iter = int(max_iter)  # max training iterations
    
    #solver_param.snapshot = iter_train  # save after an epoch
    solver_param.snapshot = 100
    solver_param.test_interval = int(iter_train)
    solver_param.test_iter[0] = int(iter_test)
    solver_param.base_lr = args.lr
    solver_param.gamma = args.lrw
    solver_param.stepsize = int(args.lrp * iter_train)
    solver_param.weight_decay = args.wd
    tmp_solver_prototxt = longPath+'tmp/%s_solver.prototxt'%net_type
    #pdb.set_trace()
    with open(tmp_solver_prototxt, 'w') as fout:
      print("hi")      
      fout.write(text_format.MessageToString(solver_param))
    # solver setup
    
    caffe.set_mode_cpu()
    
    print(">>78 train.py")
    self.solver = caffe.SGDSolver(tmp_solver_prototxt)
    # data layer setup
    #print("HERE")
    #pdb.set_trace()
    print(">>83 train py")
    layer_train = self.solver.net.layers[0]
    
    layer_test = self.solver.test_nets[0].layers[0]
    #pdb.set_trace()
    layer_train.set_batch_num(ns[0], ns[1], ns[2])#,1)
    print(">>89 train py")
    layer_test.set_batch_num(ns[0], ns[1], ns[2])#,1)
    layer_train.set_data_queue(queue_train)
    layer_test.set_data_queue(queue_test)
    
    # copy init weight if any
    if args.weight:
      self.solver.net.copy_from(args.weight)
      self.solver.test_nets[0].copy_from(args.weight)
     # print(args.weight)
    # start batcher
    batcher_train.start()
    batcher_test.start()
    #pdb.set_trace()
    def cleanup():
      batcher_train.terminate()
      batcher_test.terminate()
      batcher_train.join()
      batcher_test.join()
    import atexit
    atexit.register(cleanup)

  def train_model(self, snapshot_model=None):
    #pdb.set_trace()
    self.solver.solve(snapshot_model)
    # copy model
    pdb.set_trace() 
    shutil.copyfile(self.final_model,longPath+ 'model/%s.caffemodel'%self.net_type)


def init_caffe(cfg):
  np.random.seed(cfg.RNG_SEED)
  caffe.set_random_seed(cfg.RNG_SEED)
  if cfg.GPU_ID < 0:
    caffe.set_mode_cpu()
  else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, default=0, help='gpu id to use, -1 for cpu')
  parser.add_argument('--net', type=str, default='p', help='net type, p, r, o')
  parser.add_argument('--size', type=int, default=128, help='base batch size')
  parser.add_argument('--epoch', type=int, default=20, help='train epoches')
  parser.add_argument('--snapshot', type=str, default=None, help='snapshot model')
  parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
  parser.add_argument('--lrw', type=float, default=0.1, help='lr decay rate')
  parser.add_argument('--lrp', type=int, default=2, help='number of epoches to decay the lr')
  parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
  parser.add_argument('--weight', type=str, default=None, help='init weight for the model')
  args = parser.parse_args()

  print (args)

  net_type = args.net
  # check args
  assert net_type in ['p', 'r', 'o'], "net should be 'p', 'r', 'o'"
  cfg.NET_TYPE = net_type
  cfg.GPU_ID = args.gpu
  init_caffe(cfg)

  solver_prototxt = 'proto/%s_solver.prototxt'%net_type
  solver = Solver(solver_prototxt, args)
  solver.train_model(args.snapshot)
