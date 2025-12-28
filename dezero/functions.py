import numpy as np
from dezero.core import as_array
from dezero.core import as_variable
from dezero.core import Function
from dezero.core import log
from dezero import utils

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

class Reshape(Function):
	def __init__(self, shape):
		self.shape = shape

	def forward(self, x):
		self.x_shape = x.shape
		y = x.reshape(self.shape)
		return y
	
	def backward(self, gy):
		return reshape(gy, self.x_shape)
	
def reshape(x, shape):
	# if x.shape == shape:		# step38: 	这两步正向传播时不会有影响，但是反向传播因为是直接变成 Variable 类型变量的，
	# 	return as_variable(x)	# 			并没有调用 Function 的 __call__所以会没有 creator，导致反向传播出错
	return Reshape(shape)(x)


# class Transpose(Function):	# step38: 不支持轴transpose的代码
# 	def forward(self, x):
# 		y = np.transpose(x)
# 		return y
#
# 	def backward(self, gy):
# 		return transpose(gy)
class Transpose(Function):		# step38: 支持轴transpose的代码 如 0,1,2,3 -> 1,0,3,2
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)
	
def transpose(x):
	return Transpose()(x)


class Sum(Function):
	def __init__(self, axis, keepdims):
		self.axis = axis
		self.keepdims = keepdims
	
	def forward(self, x):
		self.x_shape = x.shape
		return x.sum(axis = self.axis, keepdims = self.keepdims)						# 这里其实是 np.sum()
	
	def backward(self, gy):
		gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) 	# 因为使用 axis 和 keepdims 会出现改变梯度形状的情况，所以要修正
		return broadcast_to(gy, self.x_shape)

def sum(x, axis = None, keepdims = False):
	return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
	def __init__(self, shape):
		self.shape = shape
	
	def forward(self, x):
		self.x_shape = x.shape
		y = np.broadcast_to(x, self.shape)
		return y
	
	def backward(self, gy):
		return sum_to(gy, self.x_shape)
	
def broadcast_to(x, shape):
	# if x.shape == shape:		# step38: 	这两步正向传播时不会有影响，但是反向传播因为是直接变成 Variable 类型变量的，
	# 	return as_variable(x)	# 			并没有调用 Function 的 __call__所以会没有 creator，导致反向传播出错
	return BroadcastTo(shape)(x)


class SumTo(Function):
	def __init__(self, shape):
		self.shape = shape
	
	def forward(self, x):
		self.x_shape = x.shape
		return utils.sum_to(x, self.shape)
	
	def backward(self, gy):
		return broadcast_to(gy, self.x_shape)

def sum_to(x, shape):
	# if x.shape == shape:		# step38: 	这两步正向传播时不会有影响，但是反向传播因为是直接变成 Variable 类型变量的，
	# 	return as_variable(x)	# 			并没有调用 Function 的 __call__所以会没有 creator，导致反向传播出错
	return SumTo(shape)(x)


class MatMul(Function):
	def forward(self, x, W):
		return np.dot(x,W)
	
	def backward(self, gy):
		x,W = self.inputs
		gx = matmul(gy, W.T)
		gW = matmul(x.T, gy)
		return gx, gW
def matmul(x, W):
	return MatMul()(x,W)

class MeanSquaredError(Function):
	def forward(self, x0, x1):
		diff = x0 - x1
		y = (diff**2).sum() / len(diff.size)		
		return y
	def backward(self, gy):
		x0, x1 = self.inputs
		gx0 = 2 * gy * (x0 - x1) / len(x0.size)
		gx1 = -gx0
		return gx0, gx1
def mean_squared_error(x0, x1):
	return MeanSquaredError()(x0, x1)