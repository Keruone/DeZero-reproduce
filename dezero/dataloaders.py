import math
import random
import numpy as np
from dezero import cuda

class DataLoader:
	def __init__(self, dataset, batch_size, shuffle = True, gpu = False):	#* Step 52: Support Cupy with cuda: New arg `gpu`
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.data_size = len(dataset)
		self.max_iter = math.ceil(self.data_size / batch_size)
		self.gpu = gpu														#* Step 52: Support Cupy with cuda: New line
		self.reset()
		
	def reset(self):
		self.iteration = 0
		if self.shuffle:
			self.index = np.random.permutation(self.data_size)
		else:
			self.index = np.arange(self.data_size)
	
	def __iter__(self):
		return self
	
	def __next__(self):
		if self.iteration == self.max_iter:
			self.reset()
			raise StopIteration
		
		i, batch_size = self.iteration, self.batch_size
		batch_index = self.index[i * batch_size : (i + 1) * batch_size]
		batch = [self.dataset[i] for i in batch_index]
		xp = cuda.cupy if self.gpu else np									#* Step 52: Support Cupy with cuda: New line
		x = xp.array([example[0] for example in batch])						#* Step 52: Support Cupy with cuda: Change `np` to `xp`
		t = xp.array([example[1] for example in batch])						#* Step 52: Support Cupy with cuda: Change `np` to `xp`

		self.iteration += 1
		return x, t
	
	def to_cpu(self):		#* Step 52: Support Cupy with cuda: New function
		self.gpu = False
	
	def to_gpu(self):		#* Step 52: Support Cupy with cuda: New function
		self.gpu = True

	def next(self):
		return self.__next__()