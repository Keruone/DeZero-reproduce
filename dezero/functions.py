import numpy as np
from dezero.core import as_array
from dezero.core import Function
from dezero.core import log

class Sin(Function):
	def forward(self, x):
		return np.sin(x)
	
	def backward(self, gy):
		x, = self.inputs
		gx = cos(x) * gy
		return gx

def sin(x):
	return Sin()(x)

class Cos(Function):
	def forward(self, x):
		return np.cos(x)
	
	def backward(self, gy):
		x, = self.inputs
		gx = -sin(x) * gy
		return gx

def cos(x):
	return Cos()(x)

class Tanh(Function):
	def forward(self, x):
		return np.tanh(x)
	
	def backward(self, gy):	# tan(x)的求导是 1-y^2，所以提取outputs
		y = self.outputs[0]()	# output 是 weakref，所以要加()
		gx = gy * (1 - y * y)
		return gx

def tanh(x):
	return Tanh()(x)
