import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
from convolution_lstm import ConvLSTM
from evaluate import Handler


# constant variables:
H = 40 # height
W = 40 # width
N = 16 # number of days in training set
path_input = "../data2/mat%02d.npy"
path_output = "../targets/tar%02d.npy"
path_model = "../models/ckpt_%06d.pth.tar"

def m_reshape(m):
	"""reshape to (CH, H, W)"""
	n = np.zeros((2, H, W))
	for i in range(0, H):
		for j in range(0, W):
			n[0][i][j] = m[i][j][0]
			n[1][i][j] = m[i][j][1]
	return n


def normalization(data):
	_range = np.max(data) - np.min(data)
	if _range > 0:
		return (data - np.min(data)) / _range
	else:
		print("# Exception in normalization method: %f %f" % (np.max(data), np.min(data)))
		return np.ones((2, H, W)) / 2


def parse_data(x, y):
	tmp = x[:, :, 1:3]
	tmp = m_reshape(tmp)
	tmp = normalization(tmp)
	inp = torch.zeros(1, 2, H, W)
	inp[0] = torch.from_numpy(tmp)
	tar = torch.zeros(1, H, W)
	tar[0] = torch.from_numpy(y)
	return inp, tar


# test set: 1, 2, 3, 4
class CLSTM(object):
	def __init__(self):
		self.trset = range(1, 17)  # a list of days in train_set
		self.index = 0  # index of "trset"
		self.net = ConvLSTM(input_channels=2, hidden_channels=[4, 8, 8, 4], kernel_size=5, step=9,
			effective_step=[8]).cuda()
		self.loss_fn = nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
	
	def next_day(self, day):
		"""determine the next day in training/testing"""
		if day == 0:
			# training
			tmp = self.trset[self.index]
			self.index = (self.index + 1) % N
		else:
			# testing
			tmp = day
			self.index = day
		#pathx = "D:\\Projects\\data2\\mat%02d.npy" % tmp
		#pathy = "..\\chengdu2\\targets\\tar%02d.npy" % tmp
		pathx = path_input % tmp
		pathy = path_output % tmp
		self.inputs = np.load(pathx).astype(float)
		self.targets = np.load(pathy).astype(float)
		#print("# Shift to day %d" % tmp)

	def get_data(self, index):
		inp = self.inputs[index]
		tar = self.targets[index + 1]
		inp, tar = parse_data(inp, tar)
		inp = Variable(inp).cuda()
		tar = Variable(tar).long().cuda()
		return inp, tar

	def forward_m(self, inp):
		output = self.net(inp)
		output = output[0][0].double()
		return output  # (bsize, 4, H, W)

	def train(self):
		for i in range(0, 143):
			inp, tar = self.get_data(i)
			self.optimizer.zero_grad()
			out = self.forward_m(inp)
			loss = self.loss_fn(out, tar)
			loss.backward()
			self.optimizer.step()
			if i == 107:
				print(loss.detach())
				#print("# Day %d" % self.index)

	def save_model(self, step):
		torch.save({'epoch': step+1, 'state_dict': self.net.state_dict(), 'optimizer': self.optimizer.state_dict()},
			path_model % (step + 1))
			#".\\model6\\ckpt_%06d.pth.tar" % (step + 1))
		print("# Model of step %d saved." % (step + 1))

	def restore_from_file(self, ckpt):
		#file_name = ".\\model6\\ckpt_%06d.pth.tar" % ckpt
		file_name = path_model % ckpt
		model_CKPT = torch.load(file_name)
		self.net.load_state_dict(model_CKPT['state_dict'])
		print("# state_dict of ConvLSTM loaded.")
		self.optimizer.load_state_dict(model_CKPT['optimizer'])

	def test(self, hd):
		layer = nn.Softmax2d()
		for i in range(0, 143):
			inp, tar = self.get_data(i)
			#self.optimizer.zero_grad()  # clear grad for optim
			out = self.forward_m(inp)
			loss = self.loss_fn(out, tar)
			#loss.backward()  # backward
			#self.optimizer.step()
			output = layer(out)
			output = torch.argmax(output, dim=1)
			hd.one_step(tar[0], output[0])
			if i % 12 == 0:
				day = self.index
				print("# Day = %d, Hour = %d" % (day, i / 6))
		return hd


	def test_graph_2(self, cks):
		"""output the predicted result"""
		# param cks is a list containing the hours you want to output
		path_out = ".\\test2\\model4out_%02d%02d.npy"
		path_tar = ".\\test2\\model4tar_%02d%02d.npy"
		layer = nn.Softmax2d()
		day = self.index
		for i in range(0, 143):
			inp, tar = self.get_data(i)
			out = self.forward_m(inp)
			loss = self.loss_fn(out, tar)
			h = i / 6
			if h in cks:
				output = layer(out)
				output = torch.argmax(output, dim=1)
				output = output.detach().cpu()
				path = path_out % (day, h)
				np.save(path, output)
				tar = tar.detach().cpu()
				path = path_tar % (day, h)
				np.save(path, tar)
				print("# Day = %d, Hour = %d" % (day, i / 6))
		print("# Finish day %d" % day)


def main(ckpt, max_step):
	"""param: start_step, end_step"""
	model = CLSTM()
	if ckpt > 0:
		model.restore_from_file(ckpt)
	for i in range(ckpt, max_step):
		for d in range(0, N):
			model.next_day(0)
			model.train()
		if i % 5 == 1:
			model.save_model(i)
	print("# Training finished.")


def main2(ckpt):
	model = CLSTM()
	model.restore_from_file(ckpt)
	hd = Handler()
	for d in range(17, 21):
		model.next_day(d)
		hd = model.test(hd)
	hd.get_metrics()


if __name__ == '__main__':
	# choose main() for training
	# choose main2() for testing
	main(0, 100)
