import numpy as np

# TP FN
# FP TN
class Handler:
	def __init__(self):
		self.matrix = np.zeros((2,2), dtype=int)

	def one_step(self, y_true, y_pred):
		tmp = np.zeros((2,2), dtype=int)
		for i in range(0, 40):
			for j in range(0, 40):
				yt = y_true[i][j]
				yp = y_pred[i][j]
				if yt > 0 and yp > 0:
					tmp[0][0] += 1
				elif yt > 0 and yp == 0:
					tmp[0][1] += 1
				elif yt == 0 and yp > 0:
					tmp[1][0] += 1
				else:
					tmp[1][1] += 1
		self.matrix = self.matrix + tmp

	def get_metrics(self):
		print(self.matrix)
		TP = self.matrix[0][0]
		FN = self.matrix[0][1]
		FP = self.matrix[1][0]
		p = TP / (TP + FP)
		r = TP / (TP + FN)
		f = 2 * p * r / (p + r)
		print(p, r, f)

	def clear_matrix(self):
		self.matrix = np.zeros((2,2), dtype=int)

	def write_metrics(self):
		TP = self.matrix[0][0]
		FN = self.matrix[0][1]
		FP = self.matrix[1][0]
		p = TP / (TP + FP)
		r = TP / (TP + FN)
		f = 2 * p * r / (p + r)
		return "%.6f,%.6f,%.6f" % (p, r, f)