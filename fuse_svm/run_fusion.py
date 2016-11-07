import os,sys
import pickle
import numpy as np
from sklearn import svm, preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

# obj_results_file = 'data/S1_train_obj_results.pkl'
# verb_results_file = 'data/S1_train_verb_results.pkl'
#obj_results_file = 'data/S1_train_obj_fc7.pkl'
#verb_results_file = 'data/S1_train_verb_fc7.pkl'
#obj_results_file = 'data/s1_train_cropped_obj_results.pkl'
#verb_results_file = 'data/S1_train_verb_results.pkl'
obj_results_file = 'data/s1_train_cropped_obj_fc7.pkl'
verb_results_file = 'data/S1_train_verb_fc7.pkl'

action_id_file = 'data/action_ids.txt'

obj_results = pickle.load(open(obj_results_file, 'r'))
verb_results = pickle.load(open(verb_results_file, 'r'))

action_ids = dict()
with open(action_id_file, 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		action_ids[tokens[0]] = int(tokens[1])

if sorted(obj_results.keys()) != sorted(verb_results.keys()):
    print 'folder keys are not equal!'
    sys.exit(0)

n_samples = len(obj_results)
n_features = obj_results[obj_results.keys()[0]][0].shape[0] + \
    verb_results[verb_results.keys()[0]][0].shape[0]
features = np.zeros((n_samples, n_features))
labels = np.zeros(n_samples, np.int32)

print 'features:', features.shape

for i, video_folder in enumerate(obj_results):
    features[i,:] = np.concatenate((obj_results[video_folder][0], verb_results[video_folder][0]))
    # verb_label = verb_results[video_folder][1]
    # obj_label = obj_results[video_folder][1]
    labels[i] = action_ids['_'.join(video_folder.split('_')[1:-1])]

scaler = preprocessing.MinMaxScaler(feature_range = (0, 1.0))
scaler.fit_transform(features)
pickle.dump(scaler, open('svm_scaler.pkl', 'w'))
features = scaler.transform(features)

pca = PCA(n_components = 300)
pca.fit(features)
pickle.dump(pca, open('svm_pca.pkl', 'w'))
features = pca.transform(features)

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
pickle.dump(svm_clf, open('svm_clf.pkl', 'w'))

# rf_clf = RandomForestClassifier(n_estimators = 30, max_depth = 100)
# print rf_clf
# rf_clf.fit(features, labels)
# pickle.dump(rf_clf, open('rf_clf.pkl', 'w'))

