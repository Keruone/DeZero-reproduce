import weakref
import numpy as np
import dezero.functions as F
from dezero.core import Parameter

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

	def _init_W(self):
		I, O = self.in_size, self.out_size
		self.W.data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)

	def forward(self, x):
		# 如果没初始化 W， 那么在传播时初始化
		if self.W.data is None:
			self.in_size = x.shape[1]	# 不能 size，size表示全部的元素个数 shape第一个元素表示的是行数，我们要的是第二个元素，表示列数
			self._init_W()
		return F.linear(x, self.W, self.b)