# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import random
from sklearn.svm import SVC
import pandas as pd
import seaborn as sn



################## Adaboost Classifier ########################


def adaboost(X, Y, testX, testY, N):
	# print len(X)

	w = (1.0/len(X))*np.ones(len(X))
	
	print w 

	models = []
	alphas= []
	flip = []

	for i in range(N):
		model = DecisionTreeClassifier(max_depth=2, max_leaf_nodes=5)
		model1 = model.fit(X,Y,sample_weight=np.array(w))
		models.append(model1)
		preds = model1.predict(X)

		# print Y, preds
		err = 0
		for j in range(len(X)):
			if(Y[j]!=preds[j]):
				err+= w[j]

		err = float(err)/np.sum(w)
		
		print err

		# if(err>0.5):
		# 	w = -1*w
		# 	for j in range(len(X)):
		# 		if(Y[j]!=preds[j]):
		# 			err+= w[j]

		# 	err = float(err)/np.sum(w)
				

		alpha = 0.5*np.log((1-err)/err) + 0.5*np.log(np.max(Y)-1)

		print alpha

		for j in range(len(X)):
			if(Y[j]==preds[j]):
				w[j] *= np.exp(-alpha)
			else:
				w[j] *= np.exp(alpha)

		w = w/np.sum(w)
		alphas.append(alpha)
	return models, alphas


def prediction(testX, testY, alphas, models):	
	predictions = []
	accuracies = []

	print alphas

	C = np.zeros((len(testY), 26))


	for i in range(len(models)):
		# print np.asarray(models[i].predict(testX))
		p = np.asarray(models[i].predict(testX))

		preds = alphas[i]*p
		# print len(preds)
		
		for j in range(len(p)):
			C[j][p[j]]+= alphas[i]


		predictions.append(preds)


	predictions = np.sum(predictions, axis=0)
	predictions = predictions.astype('int')
	count = 0

	# print preds
	# print testY 

	# print predictions, len(predictions), len(testX)
	for j in range(len(testX)):
		if(testY[j]==np.argmax(C[j])):
			count+=1
	print float(count)/len(testX)
	# count = 0

	# for j in range(len(testX)):
	# 	# print testY[j], predictions[j]

	# 	if(testY[j]==np.absolute(predictions[j])):
	# 		count+=1

	# print count 

	accuracies.append(float(count)/len(testX))
	return predictions, accuracies
	