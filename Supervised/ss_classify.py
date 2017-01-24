#!/bin/python

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression

	######
	from sklearn.model_selection import GridSearchCV
	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
	# clf = GridSearchCV(LogisticRegression(penalty='l1'), param_grid)
	# clf1 = GridSearchCV(cv=None,
 #       estimator=LogisticRegression(intercept_scaling=1, dual=False, fit_intercept=True,
 #          penalty='l2', tol=0.0001),
 #       param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001], 'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'] })

	# clf.fit(X, y)
	# clf1.fit(X, y)
	# print "-----------------------------------------------------------------"
	# print clf.best_params_
	# print "-----------------------------------------------------------------"
	# print clf1.best_params_
	# print "-----------------------------------------------------------------"

	######
	cls = LogisticRegression(penalty = 'l2', C = 10, tol=0.01)
	cls.fit(X, y)
	return cls



def addUnlabeled(speech.trainX, speech.trainy):
	

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print "  Accuracy", acc
