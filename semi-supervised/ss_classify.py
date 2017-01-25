#!/bin/python

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression

	######
	from sklearn.model_selection import GridSearchCV
	# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
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



def train_naive_bayes(X, y):
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.model_selection import GridSearchCV
	# clf1 = GridSearchCV(cv=None,
	# 	estimator=MultinomialNB(),
	# 	param_grid={'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
	# clf1.fit(X, y)
	# print "-----------------------------------------------------------------"
	# print clf1.best_params_
	# print "-----------------------------------------------------------------"

	cls = MultinomialNB(alpha = 0.01)
	cls.fit(X,y)
	return cls


def train_svm(X, y):
	from sklearn.svm import LinearSVC
	from sklearn.model_selection import GridSearchCV
	from sklearn.calibration import CalibratedClassifierCV
	# clf1 = GridSearchCV(cv=None,
	# 	estimator=LinearSVC(intercept_scaling=1, dual=False, fit_intercept=True,
	# 		penalty='l2', tol=0.0001),
	# 	param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'tol': [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.0000001], 'penalty': ['l1', 'l2']})
	# clf1.fit(X, y)
	# print "-----------------------------------------------------------------"
	# print clf1.best_params_
	# print "-----------------------------------------------------------------"

	svm = LinearSVC()
	cls = CalibratedClassifierCV(svm, method = 'isotonic')
	cls.fit(X,y)
	return cls


def addUnlabeled(unlabeled, X,y, cls_lr, cls_nb, cls_svm, speech):
	yp_lr = cls_lr.predict(unlabeled.X)
	yp_nb = cls_nb.predict(unlabeled.X)
	yp_svm = cls_svm.predict(unlabeled.X)

	pr_proba_lr = cls_lr.predict_proba(unlabeled.X)
	pr_proba_nb = cls_nb.predict_proba(unlabeled.X)
	pr_proba_svm = cls_svm.predict_proba(unlabeled.X)

	# print cls_lr.classes_
	# print "------------------------------" * 5
	# print pr_proba_lr
	# print "------------------------------" * 5
	# print pr_proba_nb
	# print "------------------------------" * 5

	# print pr_proba_svm
	# print "------------------------------" * 5

	# print speech.le.classes_

	labels_lr = speech.le.inverse_transform(yp_lr)
	labels_nb = speech.le.inverse_transform(yp_nb)
	labels_svm = speech.le.inverse_transform(yp_svm)


	# for i in range(len(labels_lr)):
	# 	print str(labels_lr[i]) + " : " + str(pr_proba_lr[i])

	# print "------------------------------" * 5
	# print labels_nb

	# f = open("data/labels.tsv", 'w')
	# f.write("FileIndex,Category\n")
	train_data, train_fnames, train_labels = getLabeledData()

	for i in xrange(len(unlabeled.fnames)):
		fname = unlabeled.fnames[i]
		maxVal_lr = max(pr_proba_lr[i])
		maxIndex_lr = pr_proba_lr[i].tolist().index(max(pr_proba_lr[i]))
		maxVal_nb = max(pr_proba_nb[i])
		maxIndex_nb = pr_proba_nb[i].tolist().index(max(pr_proba_nb[i]))
		maxVal_svm = max(pr_proba_svm[i])
		maxIndex_svm = pr_proba_svm[i].tolist().index(max(pr_proba_svm[i]))

		if (labels_lr[i] == labels_nb[i] == labels_svm[i]) and (maxVal_lr > 0.81) and (maxVal_nb > 0.79) and (maxVal_svm > 0.81):
			print "Here"
			train_fnames.append(fname)
			# train_labels.append(labels_lr[i])
			train_labels.append(speech.le.classes_[maxIndex_lr])
			train_data.append(unlabeled.data[i])
			# f.write(unlabeled.fnames[i] + "\t" + labels_lr[i])
			# f.write("\n")
	# f.close()
	from sklearn.feature_extraction.text import TfidfVectorizer

	# speech.count_vect = TfidfVectorizer(analyzer = 'word', norm = 'l2', sublinear_tf = True)  
	# speech.trainX = speech.count_vect.fit_transform(train_data)
	# #print "Speech.trainX" + str(speech.trainX)
	# from sklearn import preprocessing
	# speech.le = preprocessing.LabelEncoder()
	# speech.le.fit(speech.train_labels)
	# speech.target_labels = speech.le.classes_
	# speech.trainy = speech.le.transform(train_labels)
	
	# cls = train_classifier(speech.trainX, speech.trainy)
	# print "After addin unlabeled data to the training set"
	# evaluate(speech.trainX, speech.trainy, cls)

	count_vect = TfidfVectorizer(analyzer = 'word', norm = 'l2', sublinear_tf = True, max_features = 7916)  
	trainX = count_vect.fit_transform(train_data)
	#print "Speech.trainX" + str(speech.trainX)
	from sklearn import preprocessing
	le = preprocessing.LabelEncoder()
	le.fit(train_labels)
	target_labels = le.classes_
	trainy = speech.le.transform(train_labels)
	
	cls = train_classifier(trainX, trainy)
	print "After adding unlabeled data to the training set"
	evaluate(trainX, trainy, cls)
	return cls


def getLabeledData():
	import tarfile
	tarfname = "data/speech.tar.gz"
	tar = tarfile.open(tarfname, "r:gz")
	data, fnames, labels = read_tsv(tar, "train.tsv")
	tar.close()

	return data, fnames, labels


def read_tsv(tar, fname):
	member = tar.getmember(fname)
	print member.name
	tf = tar.extractfile(member)
	data = []
	labels = []
	fnames = []
	for line in tf:
		(ifname,label) = line.strip().split("\t")
		#print ifname, ":", label
		content = read_instance(tar, ifname)
		labels.append(label)
		fnames.append(ifname)
		data.append(content)
	return data, fnames, labels


def read_instance(tar, ifname):
	inst = tar.getmember(ifname)
	ifile = tar.extractfile(inst)
	content = ifile.read().strip()
	return content

def evaluate(X, yt, cls):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	yp = cls.predict(X)
	acc = metrics.accuracy_score(yt, yp)
	print "  Accuracy", acc
