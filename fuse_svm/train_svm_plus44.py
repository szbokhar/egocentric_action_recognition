import os,sys
import pickle
import numpy as np
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

output_dir = 'SVM_model'
features_dir = 'PLUS44_features'
obj_results_file = os.path.join(features_dir, 'AHMAD_train_3000_fc8.pkl')
verb_results_file = os.path.join(features_dir, 'ahmad_train_3000_fc8.pkl')
action_id_file = '/home/minghuam/egocentric_action_recognition/verb_obj_joint/PLUS44_data/action_ids.txt'

obj_results = pickle.load(open(obj_results_file, 'r'))
verb_results = pickle.load(open(verb_results_file, 'r'))

action_ids = dict()
with open(action_id_file, 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		action_ids[tokens[0]] = int(tokens[1])

print len(obj_results['features'].keys()), len(verb_results['features'].keys())

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

scaler = preprocessing.MinMaxScaler(feature_range = (0, 1.0))
scaler.fit_transform(features)
pickle.dump(scaler, open(os.path.join(output_dir, 'svm_scaler.pkl'), 'w'))
features = scaler.transform(features)

# pca = PCA(n_components = 300)
# pca.fit(features)
# pickle.dump(pca, open('svm_pca.pkl', 'w'))
# features = pca.transform(features)

# C_range = np.logspace(-2, 2, 10)
# gamma_range = np.logspace(-2, 2, 10)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(labels, n_iter=5, test_size=0.2, random_state=42)
# grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
# grid.fit(features, labels)
# print("The best parameters are %s with a score verb %0.2f"
#      % (grid.best_params_, grid.best_score_))

print 'training...'
#svm_clf = svm.SVC(C = 50.0, gamma = 2.0)
#svm_clf = svm.SVC(C = 45)
#svm_clf = svm.SVC(C = 50.0, gamma = 1.0)
svm_clf = svm.SVC(C = 50.0)

print svm_clf
svm_clf.fit(features, labels)
pickle.dump(svm_clf, open(os.path.join(output_dir, 'svm_clf.pkl'), 'w'))

# rf_clf = RandomForestClassifier(n_estimators = 30, max_depth = 100)
# print rf_clf
# rf_clf.fit(features, labels)
# pickle.dump(rf_clf, open('rf_clf.pkl', 'w'))