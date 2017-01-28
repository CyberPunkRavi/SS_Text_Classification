#!/bin/python

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression

	######
	from sklearn.model_selection import GridSearchCV
	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }


	cls = LogisticRegression(penalty = 'l2', C = 10, tol=0.01)

	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print "  Accuracy", acc
