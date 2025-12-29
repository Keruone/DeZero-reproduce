import numpy as np
import dezero
from dezero.core import Function, Variable, as_array, as_variable, log
from dezero import utils, cuda

class Sin(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		return xp.sin(x)				#* Step 52: Support Cupy with cuda: change `np` to `xp`
	
	def backward(self, gy):
		x, = self.inputs
		gx = cos(x) * gy
		return gx

def sin(x):
	return Sin()(x)

class Cos(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		return xp.cos(x)				#* Step 52: Support Cupy with cuda: change `np` to `xp`
	
	def backward(self, gy):
		x, = self.inputs
		gx = -sin(x) * gy
		return gx

def cos(x):
	return Cos()(x)

class Tanh(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		return xp.tanh(x)
	
	def backward(self, gy):	# tan(x)的求导是 1-y^2，所以提取outputs
		y = self.outputs[0]()	# output 是 weakref，所以要加()
		gx = gy * (1 - y * y)
		return gx

def tanh(x):
	return Tanh()(x)

class Exp(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		return xp.exp(x)
	def backward(self, gy):
		y = self.outputs[0]()  # weakref,直接使用计算结果，少计算一次
		gx = gy * y
		return gx

def exp(x):
	return Exp()(x)

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
        xp = cuda.get_array_module(gy)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
        inv_axes = tuple(xp.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)
	
def transpose(x, axes=None):
	return Transpose(axes)(x)


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
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		y = xp.broadcast_to(x, self.shape)
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
		return x.dot(W)
	
	def backward(self, gy):
		x,W = self.inputs
		gx = matmul(gy, W.T)
		gW = matmul(x.T, gy)
		return gx, gW
def matmul(x, W):
	return MatMul()(x,W)

class Linear(Function):
	def forward(self, x, W, b):
		y = x.dot(W)
		if b is not None:
			y += b
		return y
	
	def backward(self, gy):
		x, W, b = self.inputs
		gx = matmul(gy, W.T)
		gW = matmul(x.T, gy)
		gb = None if b.data is None else sum_to(gy, b.shape)	# 这里 b要给它 sum_to 变形
		return gx, gW, gb


def linear(x, W, b = None):
	return Linear()(x, W, b)

def linear_simple(x, W, b = None):
	t = matmul(x, W)
	if b is None:
		return t
	y = t + b
	return y

class Sigmoid(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		y = 1 / (1 + xp.exp(-x))
		return y
		
	def backward(self, gy):
		y = self.outputs[0]()	# 先是列表，再是 weakref
		gx = gy * y * (1 - y)
		return gx

def sigmoid(x):
	return Sigmoid()(x)

def sigmoid_simple(x):
	x = as_variable(x)
	y = 1 / (1 + exp(-x))
	return y

class ReLU(Function):
	def forward(self, x):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		return xp.maximum(x, 0.0)
	def backward(self, gy):
		x, = self.inputs
		mask = x.data > 0
		gx = gy * mask
		return gx 

def relu(x):
	return ReLU()(x)

class GetItem(Function):
	def __init__(self, slices):	# 切片参数 需要是 list 或者 np.ndarray 
		self.slices = slices
	def forward(self, x):
		y = x[self.slices]		# np.ndarray 的奇特小特性，可以自己在python中试一试
		return y
	def backward(self, gy):
		x, = self.inputs
		f = GetItemGrad(self.slices, x.shape)
		gx = f(gy)
		return gx

def get_item(x, slices):
	return GetItem(slices)(x)

class GetItemGrad(Function):
	def __init__(self, slices, x_shape):
		self.slices = slices
		self.x_shape = x_shape
	def forward(self, gy):	# 前向传播处理的都是numpy的数据
		xp = cuda.get_array_module(gy)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		gx = xp.zeros(self.x_shape)
		if xp is np:
			np.add.at(gx, self.slices, gy)
		else:
			xp.scatter_add(gx, self.slices, gy)
		return gx
	def backward(self, x):
		return get_item(x, self.slices)

class Softmax(Function):
	def __init__(self, axis):
		self.axis = axis
	def forward(self, x):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		y = xp.exp(x)
		sum_y = y.sum(axis = self.axis, keepdims=True)
		return y / sum_y
	def backward(self, gy):
		y = self.outputs[0]()
		gx = y * gy
		sumdx = gx.sum(axis=self.axis, keepdims=True)
		gx -= y * sumdx
		return gx
def softmax(x, axis = 1):
	return Softmax(axis=axis)(x)

def softmax_simple(x, axis=1):
	x = as_variable(x)
	y = exp(x)
	sum_y = sum(x, axis=axis, keepdims=True)
	return y / sum_y

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
        y = xp.clip(x, self.x_min, self.x_max)	# np.clip 将数组的所有元素限制在 [min, max] 区间内
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

#*========================
#*		Loss Func
#*========================

class MeanSquaredError(Function):
	def forward(self, x0, x1):
		diff = x0 - x1
		y = (diff**2).sum() / diff.size		
		return y
	def backward(self, gy):
		x0, x1 = self.inputs
		gx0 = 2 * gy * (x0 - x1) / x0.size
		gx1 = -gx0
		return gx0, gx1
def mean_squared_error(x0, x1):
	return MeanSquaredError()(x0, x1)

def softmax_cross_entropy_simple(x, t):	# 此处t不是one-hot
	xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
	x, t = as_variable(x), as_variable(t)
	N = x.shape[0]
	p = softmax(x)
	# print(type(p.data))
	p = clip(p, 1e-15, 1.0)  # To avoid log(0)
	log_p = log(p)
	tlog_p = log_p[xp.arange(N), t.data]
	y = -1 * sum(tlog_p) / N
	return y

class SoftmaxCrossEntropy(Function):
	def forward(self, x, t):
		xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		N = x.shape[0]
		log_z = utils.logsumexp(x, axis=1)	# 计算呢 log(sum(exp(x))), 即公式的分母
		log_p = x - log_z					# 分子先exp再log等于本身，然后使用log(a/b) = loga - logb
		log_p = log_p[xp.arange(N), t.ravel()]
		y = -log_p.sum() / xp.float32(N)
		return y
	def backward(self, gy):
		xp = cuda.get_array_module(gy)	#* Step 52: Support Cupy with cuda: Use cuda function to checkout is `np` or `cp`
		x, t = self.inputs
		N, CLS_NUM = x.shape
		t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]	# 跟前文get_item 同样的特性
		y = softmax(x)
		gx = (y - t_onehot) / N
		return gx
def softmax_cross_entropy(x, t):
	return SoftmaxCrossEntropy()(x, t)

#*========================
#*		dropout
#*========================
def dropout(x, dropout_ration = 0.5):
	x = as_variable(x)
	if dezero.Config.train:
		xp = cuda.get_array_module(x)
		mask = xp.random.rand(*x.shape) > dropout_ration
		scale = xp.array(1.0 - dropout_ration).astype(x.dtype)
		y = x * mask / scale
		return y
	else:
		return x

#*========================
#*		accuracy
#*========================
# def accuracy(y, t):
# 	y, t = as_variable(y), as_variable(t)
# 	pred = y.data.argmax(axis = 1).reshape(t.shape)	# 确保形状一致
# 	result = (pred == t.data)	# bool 类型
# 	acc = result.mean()
# 	return Variable(as_array(acc))
def accuracy(y, t):
    """
    [WAR] This function is not differentiable.
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from dezero.functions_conv import conv2d
from dezero.functions_conv import deconv2d
from dezero.functions_conv import conv2d_simple
from dezero.functions_conv import im2col
from dezero.functions_conv import col2im
from dezero.functions_conv import pooling_simple
from dezero.functions_conv import pooling
from dezero.functions_conv import average_pooling
from dezero.core import add
from dezero.core import sub
from dezero.core import rsub
from dezero.core import mul
from dezero.core import div
from dezero.core import neg
from dezero.core import pow

