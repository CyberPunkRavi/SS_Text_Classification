Semi - Supervised Text Classification

This will train a semi - supervised classifier, and save the output predictions in data/speech-pred.csv.
- The unlabeled data is Classified using multiple classifiers and the most accurate predictions are used to expand the training data set.
- The expanded training data is used to train the classifier for semi supervised classification of the Presidential Speech data. 

speech.py: All the I/O related functionality.

classify.py: To train and evaluate the classifier.
