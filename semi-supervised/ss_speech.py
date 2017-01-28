#!/bin/python

def read_files(tarfname):
	"""Read the training and development data from the speech tar file.
	The returned object contains various fields that store the data, such as:

	train_data,dev_data: array of documents (array of words)
	train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
	train_labels,dev_labels: the true string label for each document (same length as data)

	The data is also preprocessed for use with scikit-learn, as:

	count_vec: CountVectorizer used to process the data (for reapplication on new data)
	trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
	le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
	target_labels: List of labels (same order as used in le)
	trainy,devy: array of int labels, one for each document
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	speech = Data()
	print "-- train data"
	speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(tar, "train.tsv")
	print len(speech.train_data)
	print "-- dev data"
	speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(tar, "dev.tsv")
	print len(speech.dev_data)
	print "-- transforming data and labels"
	from sklearn.feature_extraction.text import TfidfVectorizer
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.feature_extraction import DictVectorizer

	speech.count_vect = TfidfVectorizer(analyzer = 'word', norm = 'l2', sublinear_tf = True)  #TfidfVectorizer(analyzer = 'word', stop_words = 'english')
	speech.trainX = speech.count_vect.fit_transform(speech.train_data)
	#print "Speech.trainX" + str(speech.trainX)
	speech.devX = speech.count_vect.transform(speech.dev_data)
	from sklearn import preprocessing
	speech.le = preprocessing.LabelEncoder()
	speech.le.fit(speech.train_labels)
	speech.target_labels = speech.le.classes_
	speech.trainy = speech.le.transform(speech.train_labels)
	speech.devy = speech.le.transform(speech.dev_labels)
	tar.close()
	return speech

def read_unlabeled(tarfname, speech):
	"""Reads the unlabeled data.

	The returned object contains three fields that represent the unlabeled data.

	data: documents, represented as sequence of words
	fnames: list of filenames, one for each document
	X: bag of word vector for each document, using the speech.vectorizer
	"""
	import tarfile
	tar = tarfile.open(tarfname, "r:gz")
	class Data: pass
	unlabeled = Data()
	unlabeled.data = []
	unlabeled.fnames = []
	for m in tar.getmembers():
		if "unlabeled" in m.name and ".txt" in m.name:
			unlabeled.fnames.append(m.name)
			unlabeled.data.append(read_instance(tar, m.name))
	unlabeled.X = speech.count_vect.transform(unlabeled.data)
	print unlabeled.X.shape
	tar.close()
	return unlabeled

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


def file_to_id(fname):
	return str(int(fname.replace("unlabeled/","").replace("labeled/","").replace(".txt","")))

def write_gold_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the truth.

	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			# iid = file_to_id(ifname)
			i += 1
			f.write(str(i))
			f.write(",")
			#f.write(ifname)
			#f.write(",")
			f.write(label)
			f.write("\n")
	f.close()

def write_basic_kaggle_file(tsvfile, outfname):
	"""Writes the output Kaggle file of the naive baseline.

	This baseline predicts OBAMA_PRIMARY2008 for all the instances.
	You will not be able to run this code, since the tsvfile is not
	accessible to you (it is the test labels).
	"""
	f = open(outfname, 'w')
	f.write("FileIndex,Category\n")
	i = 0
	with open(tsvfile, 'r') as tf:
		for line in tf:
			(ifname,label) = line.strip().split("\t")
			i += 1
			f.write(str(i))
			f.write(",")
			f.write("OBAMA_PRIMARY2008")
			f.write("\n")
	f.close()

def read_instance(tar, ifname):
	inst = tar.getmember(ifname)
	ifile = tar.extractfile(inst)
	content = ifile.read().strip()
	return content



def plot_graphs(x, y, function):
	import matplotlib.pyplot as plt
	plt.plot(x, y, color = 'blue', linewidth = 3)
	plt.axis([0, 6, 0, 20])
	plt.show()

if __name__ == "__main__":
	print "Reading data"
	tarfname = "data/speech.tar.gz"
	speech = read_files(tarfname)
   	print "Training classifier"
	import ss_classify
	cls = ss_classify.train_classifier(speech.trainX, speech.trainy)
	cls_nb = ss_classify.train_naive_bayes(speech.trainX, speech.trainy)
	cls_svm = ss_classify.train_svm(speech.trainX, speech.trainy)
	
	print "Evaluating"
	ss_classify.evaluate(speech.trainX, speech.trainy, cls)
	ss_classify.evaluate(speech.devX, speech.devy, cls)

	print "------------------------------------------------------------------------------------------"
	ss_classify.evaluate(speech.trainX, speech.trainy, cls_nb)
	ss_classify.evaluate(speech.devX, speech.devy, cls_nb)

	print "------------------------------------------------------------------------------------------"
	ss_classify.evaluate(speech.trainX, speech.trainy, cls_svm)
	ss_classify.evaluate(speech.devX, speech.devy, cls_svm)

	
	print "Reading unlabeled data"
	unlabeled = read_unlabeled(tarfname, speech)

	#Semi Supervised:
	cls_unlabeled, plot_trainX = ss_classify.addUnlabeled(unlabeled, speech.trainX, speech.trainy, cls, cls_nb, cls_svm, speech)

	# print "Testing new classifier"
#	ss_classify.evaluate(speech.devX, speech.devy, cls_unlabeled)

	# print "Writing pred file"
	# write_pred_kaggle_file(unlabeled, cls_unlabeled, "data/ss_speech-pred.csv", speech)

	# import matplotlib.pyplot as plt
	# plt.plot(speech.trainX, speech.trainy, color = 'blue', linewidth = 3)

	# You can't run this since you do not have the true labels
	# print "Writing gold file"
	# write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
	# write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")












