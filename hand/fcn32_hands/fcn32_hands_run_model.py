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


input_dir = '/home/minghuam/data/GTEA/RAW/raw_images'
output_dir = 'GTEA_output'
subject = 's2'

folders = [d for d in os.listdir(input_dir) if d.lower().startswith(subject)]
print folders

mkdir_safe(output_dir)


caffe.set_mode_gpu()

net_proto_file =  'fcn32_hands_deploy.prototxt'
model_file = 'GTEA_model/FCN32_HANDS_GTEA_S2_iter_4000.caffemodel'

net = caffe.Net(net_proto_file, model_file, caffe.TEST)
bgr_mean = [104.00699, 116.66877, 122.67892]

crop_size = 224
Imean = np.ones((crop_size, crop_size, 3), np.float32)
for i in range(3):
	Imean[:,:,i] *= bgr_mean[i]

cap = cv2.VideoCapture(0)
fourcc = cv2.cv.CV_FOURCC('M', 'P', '4', 'V')
#fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
video_writer = cv2.VideoWriter('./hand_detection.avi', fourcc, 25, (360, 203), True)

if not video_writer.isOpened():
	print 'not opened'
	sys.exit(0)

for folder in folders:
	images = ls_images(os.path.join(input_dir, folder), '.jpg')
	for img in images:
		I = cv2.resize(cv2.imread(img), (256, 256))
		x = (I.shape[1] - crop_size)/2
		y = (I.shape[0] - crop_size)/2
		I = I[y:y+crop_size, x:x+crop_size,:]
		Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
		net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
		out = net.forward()

		e_score1 = np.exp(out['left_score'])[0,:,:,:]
		e_score_sum = e_score1.sum(axis = 0)
		I1 =  e_score1[1,:,:]/e_score_sum
		I1 = (I1*255).astype(np.uint8)
		cv2.imshow('Iprob1', I1)

		e_score2 = np.exp(out['right_score'])[0,:,:,:]
		e_score_sum = e_score2.sum(axis = 0)
		I2 =  e_score2[1,:,:]/e_score_sum
		I2 = (I2*255).astype(np.uint8)
		cv2.imshow('Iprob2', I2)

		#Iout = out['score_hand'].argmax(axis = 1).astype(np.uint8)
		#Iout = np.repeat(Iout, 3, axis = 0).transpose((1, 2, 0))

		# I1 = np.repeat(I1.reshape(I1.shape + (1,)), 3, axis = 2)
		# I1[:,:,0] = 0
		# I1[:,:,1] = 0
		# I2 = np.repeat(I2.reshape(I2.shape + (1,)), 3, axis = 2)
		# I2[:,:,0] = 0
		# I2[:,:,2] = 0

		# Iret1 = cv2.addWeighted(I, 0.75, I1, 0.5, 0, 0)
		# Iret2 = cv2.addWeighted(I, 0.75, I2, 0.5, 0, 0)
		# cv2.imwrite(os.path.join(subject_output_dir, os.path.basename(pair[0])), Iret)

		#cv2.imshow('I2', Iret2)
		# cv2.imshow('Result', Iret)

		score = e_score1 + e_score2
		score_sum = score.sum(axis = 0)
		Is = score[1,:,:]/score_sum
		Is = (Is*255).astype(np.uint8)
		Is = np.repeat(Is.reshape(Is.shape + (1,)), 3, axis = 2)
		Is[:,:,0] = 0
		Is[:,:,2] = 0
		Iret = cv2.addWeighted(I, 0.75, Is, 0.5, 0, 0)

		Iret = cv2.resize(Iret, (360, 203))

		video_writer.write(Iret)

		cv2.imshow("ret", Iret)

		if cv2.waitKey(30) & 0xFF == 27:
			sys.exit(0)

video_writer.release()

