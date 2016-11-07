import os,sys,pickle
import numpy as np
import matplotlib.pyplot as plt

#results_file = 'S1_Cheese_C1_results.pkl'
#results_file = 'S1_Peanut_C1_results.pkl'
results_file = 'S1_Tea_C1_results.pkl'
results = pickle.load(open(results_file, 'r'))

images = sorted(results.keys())
predict_actions = []
predict_confidence = []
label_actions = []
for img in images:
	score = results[img][0]
	index = np.argmax(score)
	predict_actions.append(index)
	predict_confidence.append(score[0])
	label_actions.append(results[img][1])

plt.subplot(2,1,1)
plt.scatter(np.arange(len(label_actions)), label_actions, s = 1, color = 'blue')
plt.scatter(np.arange(len(predict_actions)), predict_actions, s = 1, color = 'red')
plt.subplot(2,1,2)
plt.scatter(np.arange(len(predict_confidence)), predict_confidence, s = 1, color = 'red')

plt.show()