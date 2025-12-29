import os
import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter
from dezero import cuda 

class Layer:
	def __init__(self):
		self._params = set()

	def __setattr__(self, name, value):
		if isinstance(value, (Parameter, Layer)):	# 可以查找 value 本身 或者 父类 是否满足条件
			self._params.add(name)
		#! 在 Python 中，所有类默认都会继承 object 类，不存在完全没有父类的类
		super().__setattr__(name, value)	# 所以 调用 object 类的 __setattr__

	def __call__(self, *inputs):		# *:打包
		outputs = self.forward(*inputs)	# *:解包
		if not isinstance(outputs, tuple):
			outputs = (outputs,)
		self.inputs = [weakref.ref(x) for x in inputs]		# 因为 Functions 类 已经有了强引用，所以为了避免循环引用，同时本身也确实没有必要再引用一次
		self.outputs = [weakref.ref(y) for y in outputs]
		return outputs if len(outputs) > 1 else outputs[0]
	
	def _flatten_params(self, params_dict, parent_key=""):	# 以便于扁平化的取出所有参数
		for name in self._params:
			obj = self.__dict__[name]
			key = parent_key + '/' + name if parent_key else name
			
			if isinstance(obj, Layer):
				obj._flatten_params(params_dict, parent_key=key)
			else:
				params_dict[key] = obj

	def to_cpu(self):		#* Step 52: Support Cupy with cuda: New function
		for param in self.params():
			param.to_cpu()
			
	def to_gpu(self):		#* Step 52: Support Cupy with cuda: New function
		for param in self.params():
			param.to_gpu()
			
	def forward(self, inputs):
		raise NotImplementedError()
	
	def params(self):
		for name in self._params:
			obj = self.__dict__[name]
			if isinstance(obj, Layer):
				yield from obj.params()
			else:
				yield obj	# 返回具体字典中 绑定的键值对 的 Parameters 变量实例
	
	def clear_grads(self):
		for param in self.params():
			param.clear_grad()
	
	def save_weights(self, path):
		self.to_cpu()
		params_dict = {}
		self._flatten_params(params_dict)
		array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
		try:
			np.savez_compressed(path, **array_dict)
		except (Exception, KeyboardInterrupt) as e:
			if os.path.exists(path):	# 如果保存中失败了，先删干净文件，不要将保存了一半的文件落在这
				os.ramove(path)
			raise
	
	def load_weights(self, path):
		npz = np.load(path)
		params_dict = {}
		self._flatten_params(params_dict)
		for key, param in params_dict.items():
			param.data = npz[key]

class Linear(Layer):
	def __init__(self, out_size, nobias = False, dtype = np.float32, in_size = None):
		super().__init__()
		self.in_size = in_size
		self.out_size = out_size
		self.dtype = dtype

		self.W = Parameter(None, name='W')
		if in_size is not None:
			self._init_W()
			
		if nobias :
			self.b = None
		else:
			self.b = Parameter(np.zeros(out_size, dtype=dtype), name = 'b')

	def _init_W(self, xp=np):		#* Step 52: Support Cupy with cuda: Add arg xp(default np)
		I, O = self.in_size, self.out_size
		self.W.data = xp.random.randn(I, O).astype(self.dtype) * xp.sqrt(1 / I)	#* Step 52: Support Cupy with cuda: Change `np` to `xp`

	def forward(self, x):
		# 如果没初始化 W， 那么在传播时初始化
		if self.W.data is None:
			self.in_size = x.shape[1]	# 不能 size，size表示全部的元素个数 shape第一个元素表示的是行数，我们要的是第二个元素，表示列数
			xp = cuda.get_array_module(x)	#* Step 52: Support Cupy with cuda: Create W base x
			self._init_W(xp)
		return F.linear(x, self.W, self.b)


#*==========================================================================================================================================
#*									Codes behiend here are copied from original code
#*==========================================================================================================================================

class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional convolutional layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        """Two-dimensional deconvolutional (transposed convolution)layer.

        Args:
            out_channels (int): Number of channels of output arrays.
            kernel_size (int or (int, int)): Size of filters.
            stride (int or (int, int)): Stride of filter applications.
            pad (int or (int, int)): Spatial padding width for input arrays.
            nobias (bool): If `True`, then this function does not use the bias.
            in_channels (int or None): Number of channels of input arrays. If
            `None`, parameter initialization will be deferred until the first
            forward data pass at which time the size will be determined.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


# =============================================================================
# RNN / LSTM
# =============================================================================
class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        """An Elman RNN with tanh.

        Args:
            hidden_size (int): The number of features in the hidden state.
            in_size (int): The number of features in the input. If unspecified
            or `None`, parameter initialization will be deferred until the
            first `__call__(x)` at which time the size will be determined.

        """
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self):
        self.h = None

    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()

        H, I = hidden_size, in_size
        self.x2f = Linear(H, in_size=I)
        self.x2i = Linear(H, in_size=I)
        self.x2o = Linear(H, in_size=I)
        self.x2u = Linear(H, in_size=I)
        self.h2f = Linear(H, in_size=H, nobias=True)
        self.h2i = Linear(H, in_size=H, nobias=True)
        self.h2o = Linear(H, in_size=H, nobias=True)
        self.h2u = Linear(H, in_size=H, nobias=True)
        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = (i * u)
        else:
            c_new = (f * self.c) + (i * u)

        h_new = o * F.tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new


# =============================================================================
# EmbedID / BatchNorm
# =============================================================================
class EmbedID(Layer):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.W = Parameter(np.random.randn(in_size, out_size), name='W')

    def __call__(self, x):
        y = self.W[x]
        return y


class BatchNorm(Layer):
    def __init__(self):
        super().__init__()
        # `.avg_mean` and `.avg_var` are `Parameter` objects, so they will be
        # saved to a file (using `save_weights()`).
        # But they don't need grads, so they're just used as `ndarray`.
        self.avg_mean = Parameter(None, name='avg_mean')
        self.avg_var = Parameter(None, name='avg_var')
        self.gamma = Parameter(None, name='gamma')
        self.beta = Parameter(None, name='beta')

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x):
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_nrom(x, self.gamma, self.beta, self.avg_mean.data,
                            self.avg_var.data)