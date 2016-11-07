import os,sys
import pickle
import numpy as np
from sklearn import svm

svm_dir = 'SVM_model'
features_dir = 'PLUS44_features'
obj_results_file = os.path.join(features_dir, 'AHMAD_test_3000_fc8.pkl')
verb_results_file = os.path.join(features_dir, 'ahmad_test_3000_fc8.pkl')
action_ids_file = '/home/minghuam/egocentric_action_recognition/verb_obj_joint/PLUS44_data/action_ids.txt'

obj_results = pickle.load(open(obj_results_file, 'r'))
verb_results = pickle.load(open(verb_results_file, 'r'))

if sorted(obj_results.keys()) != sorted(verb_results.keys()):
    print 'folder keys are not equal!'
    sys.exit(0)

action_ids = dict()
action_types = dict()
with open(action_ids_file, 'r') as fr:
    for line in fr.readlines():
        tokens = line.strip().split(' ')
        action_ids[tokens[0]] = int(tokens[1])
        action_types[int(tokens[1])] = tokens[0]

print len(verb_results['features'].keys()), len(obj_results['features'].keys())

'''
features = None
labels = None
for index, video in enumerate(verb_results['features']):
    if video in obj_results['features']:
        action = '_'.join(video.split('_')[1:-1])
        action_label = action_ids[action]
        verb_label = verb_results['features'][video][1]
        obj_label = obj_results['features'][video][1]
        print video, verb_label, obj_label, action_label, '{}/{}'.format(index+1, len(verb_results['features']))
        verb_features = verb_results['features'][video][0]
        obj_features = obj_results['features'][video][0]

        verb_imgs = sorted(verb_features.keys())
        verb_start_frame = int(verb_imgs[0].split('_')[-1].split('.')[0])
        verb_end_frame = int(verb_imgs[-1].split('_')[-1].split('.')[0])
        for obj_img in obj_features:
            tokens = obj_img.split('_')
            frame = int(tokens[-1].split('.')[0])
            start_frame = max(frame - 5, verb_start_frame)
            start_frame = min(start_frame, verb_end_frame)
            verb_img = '_'.join(tokens[:-1]) + '_{:06d}.jpg'.format(start_frame)
            feat = np.concatenate((verb_features[verb_img], obj_features[obj_img]))
            feat = feat.reshape((1, -1))
            if features is None:
                features = feat
            else:
                features = np.vstack((features, feat))

            if labels is None:
                labels = np.array([action_label])
            else:
                labels = np.concatenate((labels, np.array([action_label])))

        print features.shape, labels.shape

pickle.dump(features, open(os.path.join(features_dir, os.path.basename(verb_results_file).split('.')[0] + '_features.pkl'), 'w'))
pickle.dump(labels, open(os.path.join(features_dir, os.path.basename(verb_results_file).split('.')[0] + '_labels.pkl'), 'w'))

'''

features = pickle.load(open(os.path.join(features_dir, os.path.basename(verb_results_file).split('.')[0] + '_features.pkl'), 'r'))
labels = pickle.load(open(os.path.join(features_dir, os.path.basename(verb_results_file).split('.')[0] + '_labels.pkl'), 'r'))

print 'features shape:', features.shape
n_samples, n_features = features.shape[0:2]


scaler = pickle.load(open(os.path.join(svm_dir, 'svm_scaler.pkl'), 'r'))
features = scaler.transform(features)

# pca = pickle.load(open('svm_pca.pkl', 'r'))
# features = pca.transform(features)

print 'testing...'
svm_clf = pickle.load(open(os.path.join(svm_dir, 'svm_clf.pkl'), 'r'))
print svm_clf
predict_labels = svm_clf.predict(features)
print 'overall accuracy:', sum(predict_labels == labels)/float(len(labels))

# rf_clf = pickle.load(open('rf_clf.pkl', 'r'))
# print rf_clf
# predict_labels = rf_clf.predict(features)
# print sum(predict_labels == labels)/float(len(labels))

per_action_results = dict()
for i, video in enumerate(obj_results):
    action = '_'.join(video.split('_')[1:-1])
    if action not in per_action_results:
        per_action_results[action] = list()
    per_action_results[action].append((labels[i], predict_labels[i]))

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