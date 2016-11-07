caffe_root = '/home/minghuam/caffe-fcn'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import numpy as np
import argparse
import pickle

def ls_images(folder_path, extension = '.png'):
	return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)])

def mkdir_safe(path):
	if not os.path.exists(path):
		os.mkdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('test_file', help = 'Test file')
parser.add_argument('output_dir', help = 'Output directory')
args = parser.parse_args()

mkdir_safe(args.output_dir)
subject_output_dir = os.path.join(args.output_dir, os.path.basename(args.test_file).split('.')[0])
mkdir_safe(subject_output_dir)

caffe.set_mode_gpu()

net_proto_file =  'gtea_spatial_deploy.prototxt'
model_file = 'model/SPATIAL_S1_iter_4000.caffemodel'

net = caffe.Net(net_proto_file, model_file, caffe.TEST)

crop_size = 224
Imean = np.load('imagenet_mean.npy').transpose(2, 1, 0)
x = (Imean.shape[1] - crop_size)/2
y = (Imean.shape[0] - crop_size)/2
Imean = Imean[y:y+crop_size, x:x+crop_size, :]

video_images = dict()
with open(args.test_file, 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		img = tokens[0]
		label = int(tokens[1])
		video = os.path.basename(os.path.dirname(img))
		if video not in video_images:
			video_images[video] = dict()
			video_images[video]['images'] = list()
			video_images[video]['label'] = label
		video_images[video]['images'].append(img)

predict_results = dict()
scores = dict()
for video in video_images:
	label = video_images[video]['label']
	predict_labels = dict()
	score = None
	for img in video_images[video]['images']:
		I = cv2.imread(img)
		x = (I.shape[1] - crop_size)/2
		y = (I.shape[0] - crop_size)/2
		I = I[y:y+crop_size, x:x+crop_size, :]
		Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
		net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
		out = net.forward()
		# predict_label = np.argmax(out['score'].ravel())
		# if not predict_label in predict_labels:
		# 	predict_labels[predict_label] = 0
		# predict_labels[predict_label] += 1

		if score is None:
			score = net.blobs['fc7'].data.ravel()
                        #score = out['score'].ravel()
		else:
                        score += net.blobs['fc7'].data.ravel()
			#score += out['score'].ravel()
	score = score/len(video_images[video]['images'])
	scores[video] = (score, label)
	predict_label = np.argmax(score)

	# max_count = 0
	# predict_label = -1
	# for l in predict_labels:
	# 	if max_count < predict_labels[l]:
	# 		max_count = predict_labels[l]
	# 		predict_label = l

	print video, label, predict_label
	predict_results[video] = (label, predict_label)

save_file = os.path.splitext(os.path.basename(args.test_file))[0] + '_fc7_results.pkl'
pickle.dump(scores, open(os.path.join(save_file), 'w'))


# overall accuracy
correct_count = 0
for video in predict_results:
	if predict_results[video][0] == predict_results[video][1]:
		correct_count += 1
print 'overall accuracy:',float(correct_count)/len(predict_results)

per_action_results = dict()
for video in predict_results:
	action = '_'.join(video.split('_')[1:-1])
	if action not in per_action_results:
		per_action_results[action] = list()
	per_action_results[action].append(predict_results[video])

per_action_accuracy = dict()
for action in per_action_results:
	total_count = len(per_action_results[action])
	correct_count = 0
	for (label, predict_label) in per_action_results[action]:
		if label == predict_label:
			correct_count += 1
	per_action_accuracy[action] = float(correct_count)/total_count

accuracies = per_action_accuracy.values()
print 'average accuracy:', sum(accuracies)/len(accuracies)
