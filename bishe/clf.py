import numpy as np
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# constant variables
TEST = 4
TRAIN = 16

def normalization(data):
	_range = np.max(data) - np.min(data)
	return (data - np.min(data)) / _range


def get_input(day, index):
	m = np.load("..\\data2\\mat%02d.npy" % day)[index]
	n = m[:, :, 1:3]
	n = np.reshape(n, (1600, 2))
	data = normalization(n)
	return data


def get_target(day, index):
	#m = np.load(".\\data\\tar%02d.npy" % day)[index]  # (40, 40)
	m = np.load("..\\chengdu2\\targets\\tar%02d.npy" % day)[index]
	n = np.reshape(m, 1600)
	return n


def get_train_set(index):
	#trset = [2, 3, 4, 5]
	#trset = range(3, 11)
	#trset = [2, 3, 4, 5, 6, 7, 9, 10]
	trset = range(1, 17)
	d = TRAIN
	train_X = np.zeros((d * 40 * 40, 2))
	train_y = np.zeros(d * 40 * 40)
	i = 0
	for d in trset:
		tmp = get_input(d, index)
		train_X[i:i+1600, :] = tmp
		tar = get_target(d, index + 1)
		train_y[i:i+1600] = tar
	return train_X, train_y


def get_test_set(index):
	#tsset = [1]
	#tsset = [1, 2]
	#tsset = [1, 8]
	tsset = [17, 18, 19, 20]
	n = TEST * 40 * 40
	test_X = np.zeros((n, 2))
	test_y = np.zeros(n)
	i = 0
	for d in tsset:
		inp = get_input(d, index)
		test_X[i:i+1600, :] = inp
		tar = get_target(d, index + 1)
		test_y[i:i+1600] = tar
	return test_X, test_y


def evaluate(rg, y, y_):
	TP = 0
	FN = 0
	FP = 0
	TN = 0
	for i in range(0, rg):
		gt = y[i]
		pd = y_[i]
		if gt >= 1.0 and pd >= 1.0:
			TP += 1
		elif gt >= 1.0 and pd < 1.0:
			FN += 1
		elif gt < 1.0 and pd >= 1.0:
			FP += 1
		else:
			TN += 1
	return (TP, FN, FP, TN)


# different clfs:
# kNN: clf = neighbors.KNeighborsClassifier(n_neighbors=4)
# DT: clf = DecisionTreeClassifier(max_leaf_nodes=8)
# SVM: clf = svm.SVC(decision_function_shape='ovr', kernel='rbf', C=1.0)
# GBDT: clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=7)
def process():
	matrix = np.zeros(4, dtype=int)
	num_test = TEST
	for i in range(0, 143):
		X, y = get_train_set(i)
		# choose one clf from the above:
		clf = GradientBoostingClassifier(learning_rate=0.1, max_depth=7)
		clf.fit(X, y)
		X1, y1 = get_test_set(i)
		pd = clf.predict(X1)
		matrix = matrix + evaluate(40 * 40 * num_test, y1, pd)
		if i > 0 and i % 6 == 0:
			print("# Hour %d" % (i / 6))
	TP = matrix[0]
	FN = matrix[1]
	FP = matrix[2]
	TN = matrix[3]
	p = TP / (TP + FP)
	r = TP / (TP + FN)
	f = 2*p*r / (p+r)
	print(p, r, f)


if __name__ == '__main__':
	process()
