caffe_root = '/home/minghuam/caffe-dev'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import cv2
import caffe
import numpy as np

#masks_dir = 'PLUS44_data/ALIREZA_mask'
#output_file = 'PLUS44_data/ALIREZA_mean.binaryproto'
masks_dir = 'GTEA_data/S4_mask'
output_file = 'GTEA_data/S4_mean.binaryproto'

n_images = 0
Isum = np.zeros((256, 256), np.float)
for d in os.listdir(masks_dir):
	print d
	imgs = os.listdir(os.path.join(masks_dir, d))
	n_images += len(imgs)
	for img in imgs:
		I = cv2.imread(os.path.join(masks_dir, d, img), cv2.CV_LOAD_IMAGE_GRAYSCALE)
		Isum += cv2.resize(I, (256, 256))

Imean = Isum/float(n_images)
print Imean.shape

cv2.imshow('Imean', Imean.astype(np.uint8))
while cv2.waitKey(0) & 0xFF != 27:
	continue

Imean = Imean.reshape((1, 1,) + Imean.shape)

blob = caffe.proto.caffe_pb2.BlobProto()
blob.num, blob.channels, blob.height, blob.width = Imean.shape
blob.data.extend(Imean.astype(float).flat)
fw = open(output_file, 'wb')
fw.write(blob.SerializeToString())
fw.close()
