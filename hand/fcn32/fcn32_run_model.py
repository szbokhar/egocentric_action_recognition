caffe_root = '/home/minghuam/caffe-fcn'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import numpy as np
import argparse

def ls_images(folder_path, extension = '.png'):
	return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)])

def mkdir_safe(path):
	if not os.path.exists(path):
		os.mkdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('test_file', help = 'Test file')
parser.add_argument('model_file', help = 'Model file')
parser.add_argument('output_dir', help = 'Output directory')
args = parser.parse_args()

mkdir_safe(args.output_dir)
subject_output_dir = os.path.join(args.output_dir, os.path.basename(args.test_file).split('.')[0])
mkdir_safe(subject_output_dir)

caffe.set_mode_gpu()

net_proto_file =  'fcn32_deploy.prototxt'
net = caffe.Net(net_proto_file, args.model_file, caffe.TEST)
bgr_mean = [104.00699, 116.66877, 122.67892]

image_pairs = list()
with open(args.test_file, 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		image_pairs.append((tokens[0], tokens[1]))

crop_size = 224
I = cv2.imread(image_pairs[0][0])
Imean = np.ones((crop_size, crop_size, 3), np.float32)
for i in range(3):
	Imean[:,:,i] *= bgr_mean[i]

for pair in image_pairs:
	I = cv2.imread(pair[0])
	x = (I.shape[1] - crop_size)/2
	y = (I.shape[0] - crop_size)/2
	I = I[y:y+crop_size, x:x+crop_size,:]
	Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
	net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
	out = net.forward()

	e_score = np.exp(out['score'])[0,:,:,:]
	e_score_sum = e_score.sum(axis = 0)

	I1 =  e_score[1,:,:]/e_score_sum

	cv2.imshow('Iprob', (I1*255).astype(np.uint8))

	#Iout = out['score_hand'].argmax(axis = 1).astype(np.uint8)
	#Iout = np.repeat(Iout, 3, axis = 0).transpose((1, 2, 0))

	Iout = (I1*255).astype(np.uint8).reshape(I1.shape[0], I1.shape[1], 1)
	Iout = np.repeat(Iout, 3, axis = 2)
	cv2.imwrite(os.path.join(subject_output_dir, os.path.basename(pair[0])), Iout)

	Iout[:,:,0] = 0
	Iout[:,:,2] = 0
	Iret = cv2.addWeighted(I, 0.75, Iout, 0.5, 0, 0)
	#cv2.imwrite(os.path.join(subject_output_dir, os.path.basename(pair[0])), Iret)
	
	cv2.imshow('I', I)
	cv2.imshow('Result', Iret)
	if cv2.waitKey(100) & 0xFF == 27:
		break

