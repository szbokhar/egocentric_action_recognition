import os,sys
import pickle
import numpy as np
from sklearn import svm

# obj_results_file = 'data/S1_test_obj_results.pkl'
# verb_results_file = 'data/S1_test_verb_results.pkl'
#obj_results_file = 'data/S1_test_obj_fc7.pkl'
#verb_results_file = 'data/S1_test_verb_fc7.pkl'
#obj_results_file = 'data/s1_test_cropped_obj_results.pkl'
#verb_results_file = 'data/S1_test_verb_results.pkl'
obj_results_file = 'data/s1_test_cropped_obj_fc7.pkl'
verb_results_file = 'data/S1_test_verb_fc7.pkl'

action_ids_file = 'data/action_ids.txt'

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

n_samples = len(obj_results)
n_features = obj_results[obj_results.keys()[0]][0].shape[0] + \
    verb_results[verb_results.keys()[0]][0].shape[0]
features = np.zeros((n_samples, n_features))
labels = np.zeros(n_samples, np.int32)

print 'features:', features.shape

for i, video_folder in enumerate(obj_results):
    features[i,:] = np.concatenate((obj_results[video_folder][0], verb_results[video_folder][0]))
    labels[i] = action_ids['_'.join(video_folder.split('_')[1:-1])]

scaler = pickle.load(open('svm_scaler.pkl', 'r'))
features = scaler.transform(features)

pca = pickle.load(open('svm_pca.pkl', 'r'))
features = pca.transform(features)

print 'testing...'
svm_clf = pickle.load(open('svm_clf.pkl', 'r'))
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