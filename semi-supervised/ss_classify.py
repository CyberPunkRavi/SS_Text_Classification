#!/bin/python

def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains a logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression

	######
	from sklearn.model_selection import GridSearchCV

	cls = LogisticRegression(penalty = 'l2', C = 10, tol=0.01)
	cls.fit(X, y)
	return cls



def train_naive_bayes(X, y):
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.model_selection import GridSearchCV
	

	cls = MultinomialNB(alpha = 0.01)
	cls.fit(X,y)
	return cls


def train_svm(X, y):
	from sklearn.svm import LinearSVC
	from sklearn.model_selection import GridSearchCV
	from sklearn.calibration import CalibratedClassifierCV
	
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

	
	labels_lr = speech.le.inverse_transform(yp_lr)
	labels_nb = speech.le.inverse_transform(yp_nb)
	labels_svm = speech.le.inverse_transform(yp_svm)


	train_data, train_fnames, train_labels = getLabeledData()

	for i in xrange(len(unlabeled.fnames)):
		fname = unlabeled.fnames[i]
		maxVal_lr = max(pr_proba_lr[i])
		maxIndex_lr = pr_proba_lr[i].tolist().index(max(pr_proba_lr[i]))
		maxVal_nb = max(pr_proba_nb[i])
		maxIndex_nb = pr_proba_nb[i].tolist().index(max(pr_proba_nb[i]))
		maxVal_svm = max(pr_proba_svm[i])
		maxIndex_svm = pr_proba_svm[i].tolist().index(max(pr_proba_svm[i]))

		if (labels_lr[i] == labels_nb[i] == labels_svm[i]) and (maxVal_lr > 0.79) and (maxVal_nb > 0.75) and (maxVal_svm > 0.79):
			print "Here"
			train_fnames.append(fname)
			# train_labels.append(labels_lr[i])
			train_labels.append(speech.le.classes_[maxIndex_lr])
			train_data.append(unlabeled.data[i])
			# f.write(unlabeled.fnames[i] + "\t" + labels_lr[i])
			# f.write("\n")
	# f.close()
	from sklearn.feature_extraction.text import TfidfVectorizer


	count_vect = TfidfVectorizer(analyzer = 'word', norm = 'l2', sublinear_tf = True) #max_features = 7916)  

	
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



	##Added newly##
	import tarfile
	tarfname = "data/speech.tar.gz"
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []
	unlabeled.fnames = []
	for m in tar.getmembers():
		if "unlabeled" in m.name and ".txt" in m.name:
			unlabeled.fnames.append(m.name)
			unlabeled.data.append(read_instance(tar, m.name))
	unlabeled.X = count_vect.transform(unlabeled.data)
	print unlabeled.X.shape
	tar.close()


	#unlabeled = read_unlabeled(tarfname, speech)
	# X_test = count_vect.transform(unlabeled.X)
	write_pred_kaggle_file(unlabeled, cls, "data/ss_speech-pred.csv", speech)

	return cls, trainX


def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
	"""Writes the predictions in Kaggle format.

	Given the unlabeled object, classifier, outputfilename, and the speech object,
	this function write the predictions of the classifier on the unlabeled data and
	writes it to the outputfilename. The speech object is required to ensure
	consistent label names.
	"""
	yp = cls.predict(unlabeled.X)
	labels = speech.le.inverse_transform(yp)
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	for i in xrange(len(unlabeled.fnames)):
		fname = unlabeled.fnames[i]
		# iid = file_to_id(fname)
		f.write(str(i+1))
		f.write(",")
		#f.write(fname)
		#f.write(",")
		f.write(labels[i])
		f.write("\n")
	f.close()


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
