import math
import cv2
import caffe
import numpy as np

def gen_bbox(hotmap, offset, scale, th):
	h, w = hotmap.shape
	stride = 2
	win_size = 12
	hotmap = hotmap.reshape((h, w))
	keep = hotmap > th
	pos = np.where(keep)
	score = hotmap[keep]
	offset = offset[:, keep]
	x, y = pos[1], pos[0]
	x1 = stride * x
	y1 = stride * y
	x2 = x1 + win_size
	y2 = y1 + win_size
	x1 = x1 / scale
	y1 = y1 / scale
	x2 = x2 / scale
	y2 = y2 / scale
	bbox = np.vstack([x1, y1, x2, y2, score, offset]).transpose()
	return bbox.astype(np.float32)

def nms(dets, thresh, meth='Union'):
	x1 = dets[:, 0]
	y1 = dets[:, 1]
	x2 = dets[:, 2]
	y2 = dets[:, 3]
	scores = dets[:, 4]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = scores.argsort()[::-1]
	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w * h
		if meth == 'Union':
			ovr = inter / (areas[i] + areas[order[1:]] - inter)
		else:
			ovr = inter / np.minimum(areas[i], areas[order[1:]])
		inds = np.where(ovr <= thresh)[0]
		order = order[inds + 1]
	return keep

def bbox_reg(bboxes):
	w = bboxes[:, 2] - bboxes[:, 0]
	h = bboxes[:, 3] - bboxes[:, 1]
	bboxes[:, 0] += bboxes[:, 5] * w
	bboxes[:, 1] += bboxes[:, 6] * h
	bboxes[:, 2] += bboxes[:, 7] * w
	bboxes[:, 3] += bboxes[:, 8] * h
	return bboxes

def make_square(bboxes):
	x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
	y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
	w = bboxes[:, 2] - bboxes[:, 0]
	h = bboxes[:, 3] - bboxes[:, 1]
	size = np.vstack([w, h]).max(axis=0).transpose()
	bboxes[:, 0] = x_center - size / 2
	bboxes[:, 2] = x_center + size / 2
	bboxes[:, 1] = y_center - size / 2
	bboxes[:, 3] = y_center + size / 2
	return bboxes

def crop_face(img, bbox, wrap=True):
	height, width = img.shape[:-1]
	x1, y1, x2, y2 = bbox
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	if x1 >= width or y1 >= height or x2 <= 0 or y2 <= 0:
		print '[WARN] ridiculous x1, y1, x2, y2'
		return None
	if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
		# out of boundary, still crop the face
		if not wrap:
			return None
		h, w = y2 - y1, x2 - x1
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

pnet = caffe.Net('proto/p.prototxt', 'tmp/pnet_iter_446000.caffemodel', caffe.TEST)
rnet = caffe.Net('proto/r.prototxt', 'tmp/rnet_iter_116000.caffemodel', caffe.TEST)
onet = caffe.Net('proto/o.prototxt', 'tmp/onet_iter_90000.caffemodel', caffe.TEST)

img = cv2.imread('9387493245278.jpg', cv2.IMREAD_COLOR)
min_size = 24
factor = 0.709
base = 12. / min_size
height, width = img.shape[:-1]
l = min(width, height)
l *= base
scales = []
while l > 12:
	scales.append(base)
	base *= factor
	l *= factor

### pnet ###
bboxes_in_all_scales = np.zeros((0, 4 + 1 + 4), dtype=np.float32)
for scale in scales:
	w, h = int(math.ceil(scale * width)), int(math.ceil(scale * height))
	data = cv2.resize(img, (w, h))
	data = data.transpose((2, 0, 1)).astype(np.float32) # order now: ch, height, width
	data = (data - 128) / 128
	data = data.reshape((1, 3, h, w)) # order now: batch, ch, height, width
	pnet.blobs['data'].reshape(*data.shape)
	pnet.blobs['data'].data[...] = data
	pnet.forward()
	prob = pnet.blobs['prob'].data
	bbox_pred = pnet.blobs['bbox_pred'].data
	bboxes = gen_bbox(prob[0][1], bbox_pred[0], scale, 0.6)
	keep = nms(bboxes, 0.5) # nms in each scale
	bboxes = bboxes[keep]
	bboxes_in_all_scales = np.vstack([bboxes_in_all_scales, bboxes])
# nms in total
keep = nms(bboxes_in_all_scales, 0.7)
bboxes_in_all_scales = bboxes_in_all_scales[keep]
bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
bboxes_in_all_scales = make_square(bboxes_in_all_scales)
pnet_boxes = bboxes_in_all_scales.copy()
imgdraw_pnet = img.copy()
for i in range(len(pnet_boxes)):
	x1, y1, x2, y2, score = pnet_boxes[i, :5]
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	cv2.rectangle(imgdraw_pnet, (x1, y1), (x2, y2), (0, 0, 255), 2)
	cv2.putText(imgdraw_pnet, '%.03f'%score, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
cv2.imshow("pnet", imgdraw_pnet)
cv2.imwrite("pnet.jpg", imgdraw_pnet)
cv2.waitKey(0)

# maybe redundent
fake = np.zeros((1, 3, 12, 12), dtype=np.float32)
pnet.blobs['data'].reshape(*fake.shape)
pnet.blobs['data'].data[...] = fake
pnet.forward()

### rnet ###
n = len(bboxes_in_all_scales)
data = np.zeros((n, 3, 24, 24), dtype=np.float32)
for i, bbox in enumerate(bboxes_in_all_scales):
	face = crop_face(img, bbox[:4])
	data[i] = cv2.resize(face, (24, 24)).transpose((2, 0, 1))
data = (data - 128) / 128
rnet.blobs['data'].reshape(*data.shape)
rnet.blobs['data'].data[...] = data
rnet.forward()
prob = rnet.blobs['prob'].data
bbox_pred = rnet.blobs['bbox_pred'].data
prob = prob.reshape(n, 2)
bbox_pred = bbox_pred.reshape(n, 4)
keep = prob[:, 1] > 0.7
bboxes_in_all_scales = bboxes_in_all_scales[keep]
bboxes_in_all_scales[:, 4] = prob[keep, 1]
bboxes_in_all_scales[:, 5:9] = bbox_pred[keep]
keep = nms(bboxes_in_all_scales, 0.7)
bboxes_in_all_scales = bboxes_in_all_scales[keep]
bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
bboxes_in_all_scales = make_square(bboxes_in_all_scales)
rnet_boxes = bboxes_in_all_scales.copy()
imgdraw_rnet = img.copy()
for i in range(len(rnet_boxes)):
	x1, y1, x2, y2, score = rnet_boxes[i, :5]
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	cv2.rectangle(imgdraw_rnet, (x1, y1), (x2, y2), (0, 0, 255), 2)
	cv2.putText(imgdraw_rnet, '%.03f'%score, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
cv2.imshow("rnet", imgdraw_rnet)
cv2.imwrite("rnet.jpg", imgdraw_rnet)
cv2.waitKey(0)

# maybe redundent
fake = np.zeros((1, 3, 24, 24), dtype=np.float32)
rnet.blobs['data'].reshape(*fake.shape)
rnet.blobs['data'].data[...] = fake
rnet.forward()

### onet ###
n = len(bboxes_in_all_scales)
data = np.zeros((n, 3, 48, 48), dtype=np.float32)
for i, bbox in enumerate(bboxes_in_all_scales):
	face = crop_face(img, bbox[:4])
	data[i] = cv2.resize(face, (48, 48)).transpose((2, 0, 1))
data = (data - 128) / 128
onet.blobs['data'].reshape(*data.shape)
onet.blobs['data'].data[...] = data
onet.forward()
prob = onet.blobs['prob'].data
bbox_pred = onet.blobs['bbox_pred'].data
prob = prob.reshape(n, 2)
bbox_pred = bbox_pred.reshape(n, 4)
keep = prob[:, 1] > 0.4
bboxes_in_all_scales = bboxes_in_all_scales[keep]
bboxes_in_all_scales[:, 4] = prob[keep, 1]
bboxes_in_all_scales[:, 5:9] = bbox_pred[keep]
bboxes_in_all_scales = bbox_reg(bboxes_in_all_scales)
keep = nms(bboxes_in_all_scales, 0.5, 'Min')
bboxes_in_all_scales = bboxes_in_all_scales[keep]
onet_boxes = bboxes_in_all_scales.copy()
imgdraw_onet = img.copy()
for i in range(len(onet_boxes)):
	x1, y1, x2, y2, score = onet_boxes[i, :5]
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	cv2.rectangle(imgdraw_onet, (x1, y1), (x2, y2), (0, 0, 255), 2)
	cv2.putText(imgdraw_onet, '%.03f'%score, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
cv2.imshow("onet", imgdraw_onet)
cv2.imwrite("onet.jpg", imgdraw_onet)
cv2.waitKey(0)
