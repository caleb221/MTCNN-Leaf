# pylint: disable=bad-indentation, no-member, invalid-name, line-too-long
import multiprocessing
import cv2
import lmdb
import numpy as np
import functools
import pdb
from jfda.config import cfg


class MiniBatcher(multiprocessing.Process):
  '''generate minibatch
  given a queue, put (negatives, positives, part faces, landmark faces) = (n1, n2, n3, n4)
  '''

  def __init__(self, db_names, ns, net_type):
    '''order: negatives, positives, part faces, landmark faces
    '''
    super(MiniBatcher, self).__init__()
    self.ns = ns
    self.n = functools.reduce(lambda x, acc: acc + x, ns, 0)
    self._start = [0 for _ in range(3)]
    self.net_type = net_type
    self.db_names = db_names
    self.db = [lmdb.open(db_name) for db_name in db_names]
    self.tnx = [db.begin() for db in self.db]
    #pdb.set_trace()
    self.db_size = [int(tnx.get('size'.encode())) for tnx in self.tnx]

  def __del__(self):
    for tnx in self.tnx:
      tnx.abort()
    for db in self.db:
      db.close()

  def set_queue(self, queue):
    self.queue = queue

  def get_size(self):
    return self.db_size

  def _make_transform(self, data, bbox=None):
    # gray scale
    if np.random.rand() < cfg.GRAY_PROB:
      gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
      data[:, :, 0] = gray
      data[:, :, 1] = gray
      data[:, :, 2] = gray
    # flip
    if np.random.rand() < cfg.FLIP_PROB:
      data = data[:, ::-1, :]
      if bbox is not None:
        # [dx1 dy1 dx2 dy2] --> [-dx2 dy1 -dx1 dy2]
        bbox[0], bbox[2] = -bbox[2], -bbox[0]
    data = data.transpose((2, 0, 1))
    return data, bbox

  def run(self):
    intpu_size = cfg.NET_INPUT_SIZE[self.net_type]
    data_shape = (intpu_size, intpu_size, 3)
    bbox_shape = (4,)
    n = self.n
    while True:
      data = np.zeros((n, 3, intpu_size, intpu_size), dtype=np.float32)
      bbox_target = np.zeros((n, 4), dtype=np.float32)
      label = np.zeros(n, dtype=np.float32)

      start = self._start
      end = [start[i] + self.ns[i] for i in range(3)]
      for i in range(3):
        if end[i] > self.db_size[i]:
          end[i] -= self.db_size[i]
          start[i] = end[i]
          end[i] = start[i] + self.ns[i]

      idx = 0
      # negatives
      #print("START: ",start[0])
      #print("END: ",end[0])
      
      for i in range(start[0], end[0]):
        data_key = '%08d_data'%i
        
        #print(data_key)
        #
        try:
            _data = np.fromstring(self.tnx[0].get(data_key.encode()), dtype=np.uint8).reshape(data_shape)
            data[idx], _1 = self._make_transform(_data)
            idx += 1
        except:
            print("")
      # positives
      
      for i in range(start[1], end[1]):
        data_key = '%08d_data'%i
        bbox_key = '%08d_bbox'%i
        _data = np.fromstring(self.tnx[1].get(data_key), dtype=np.uint8).reshape(data_shape)
        _bbox_target = np.fromstring(self.tnx[1].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
        data[idx], bbox_target[idx] = self._make_transform(_data, _bbox_target)
        idx += 1
      # part faces
      for i in range(start[2], end[2]):
        data_key = '%08d_data'%i
        bbox_key = '%08d_bbox'%i
        _data = np.fromstring(self.tnx[2].get(data_key), dtype=np.uint8).reshape(data_shape)
        _bbox_target = np.fromstring(self.tnx[2].get(bbox_key), dtype=np.float32).reshape(bbox_shape)
        data[idx], bbox_target[idx] = self._make_transform(_data, _bbox_target)
        idx += 1
      # label
      label[:self.ns[0]] = 0
      label[self.ns[0]: self.ns[0]+self.ns[1]] = 1
      label[self.ns[0]+self.ns[1]: self.ns[0]+self.ns[1]+self.ns[2]] = 2
      label[self.ns[0]+self.ns[1]+self.ns[2]:] = 3

      self._start = end
      data = (data - 128) / 128 # simple normalization
      minibatch = {'data': data,
                   'bbox_target': bbox_target,
                   'label': label}
      self.queue.put(minibatch)
