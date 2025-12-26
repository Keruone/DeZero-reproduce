import numpy as np
import unittest
import weakref
import contextlib

class Config:
	enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
	"""Config 配置设置，搭配 with 使用

	Args:
		name (str): 要修改的Config中具体字段名称
		value (看具体配置类型): name对应的字段设置的数值
	"""
	old_value = getattr(Config, name)
	setattr(Config, name, value)
	try:
		yield
	finally:
		setattr(Config, name, old_value)

def no_grad():
	"""搭配 with，简化写法"""
	return using_config('enable_backprop', False)

class Variable:
	__array_priority__ = 200	# 设置实例的 array 运算优先级，要高于 numpy的实例，这样才可以运算
	def __init__(self, data, name = None):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{} is not supported'.format(type(data))) 

		self.data = data
		self.name = name
		self.grad = None
		self.creator = None
		self.generation = 0
	
	@property
	def shape(self):
		return self.data.shape
	
	@property
	def ndim(self):
		return self.data.ndim
	
	@property
	def size(self):
		return self.data.size
	
	@property
	def dtype(self):
		return self.data.dtype

	def __len__(self):
		return len(self.data)
	
	def __repr__(self):
		if self.data is None:
			return 'Variable(None)'
		p = str(self.data).replace('\n', '\n' + ' ' * 9)
		return 'Variable(' + p + ')'

	def set_creator(self, func):	
		"""用于指定父级，即是哪个函数计算得到的它，用于调用反向传播"""
		self.creator = func
		self.generation = func.generation + 1
		#? step15-16: 为什么不选择在 F.__call__ 中设置generation?
		#! 核心原因：职责分离
		# 符合“对象自治”原则:
		# 	Variable 知道“当我被某个函数创建时，我应该怎么做”
		# 	而不是让 Function 来告诉 Variable “你该是什么 generation”
		# 这就像：
		# 	孩子知道自己是谁的孩子（creator），并自动知道自己的辈分（generation = parent.gen + 1）
		# 	而不是父母强行给孩子贴标签
	
	def clear_grad(self):
		self.grad = None
		
	def backward(self,retain_grad = False):		# step 18: 只保留端侧数据的梯度，以减少memory使用
		"使用循环的方式实现 backward"
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = []
		seen_set = set()	#? 为什么set?  set 的成员判断效率远高于 list，且能天然保证 “元素唯一” 搜索效率：set=O(1) list=O(n)
		def add_func(f):
			if f not in seen_set:
				funcs.append(f)
				seen_set.add(f)
				funcs.sort(key= lambda x: x.generation)	# 默认从小到大排序，这样也恰好与pop符合
				#? 怎么阅读 上面的参数?
				# 	key = lambda x: x.generation
				# 	├─ key：sort的参数，指定“按什么规则排序”
				# 	└─ lambda x: x.generation：匿名函数，作为排序的“规则函数”
				# 		├─ lambda x：定义匿名函数，x是函数的输入（代表列表里的每个元素）
				# 		├─ : ：分隔“参数”和“返回值”
				# 		└─ x.generation：函数的返回值（取x的generation属性）

		add_func(self.creator)
		while funcs:
			# 每次仅处理一个函数，所以只要考虑这个函数的输出和输入
			f = funcs.pop()
			xs, ys = f.inputs, f.outputs	# step17: ys 中的y 是弱链接
			gys = [y().grad for y in ys]	# step17: 对于这个函数来说，输入是强链接，输出是弱链接，所以提取参数是要注意
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple):
				gxs = (gxs,)
			for gx, x in zip(gxs, xs):		# xs与gxs一定是可以匹配上的
				if x.grad == None:
					x.grad = gx
				else:
					x.grad = x.grad + gx	#! 这里千万不能是 x.grad += gx
					#! 1. Python 中变量赋值本质上是“名字绑定”（name binding），即让一个名字（变量）指向某个对象 —— 可以理解为“引用”。
					#! 2. 大多数运算（如 a + b）会创建一个新对象，然后赋值操作（x = ...）会让变量名重新绑定到这个新对象。
					#! 3. 原地运算符（如 +=, *=, etc.）对可变对象（如 list, np.ndarray）会尝试直接修改原对象的内容（in-place），而不改变其身份（id 不变）；对不可变对象（如 int, str），则退化为普通赋值（创建新对象）。
				if x.creator is not None:
					add_func(x.creator)
				
				if not retain_grad:
					for output in f.outputs:
						output().grad = None	# step 18 如果需要梯度就删除


def as_array(x):		# 处理标量为 np.ndarray 类型 #step 21 标量要先经过 as_array() 再经过 as_variable()
	if np.isscalar(x):
		return np.array(x)
	return x

def as_variable(obj):	#假定是np.ndarray 或 Variable
	if isinstance(obj, Variable):
		return obj
	return Variable(obj)

class Function:
	def __call__(self, *inputs):						## 修改为可接收多项 Var 输入，会打包为tuple
		inputs = [as_variable(input) for input in inputs]	# 转换输入为 Var 类型变量，一遍多type数据一起运算 #! 注意，input 不会立即销毁，因为function 有强引用，根据引用计数，一时间删不掉
		xs = [input.data for input in inputs]
		ys = self.forward(*xs)							# step12: 解包，让多个x可以与具体调用的有多个参数的函数相匹配，也是这一步，让exp square可以正常运行
		if not isinstance(ys, tuple):					# step12: 如果不是元组
			ys = (ys,)									# 			则变为元组，确保可迭代
		outputs = [Variable(as_array(y)) for y in ys]	# 调用as_array,确保创建的是var 参数是np.ndarry
		if Config.enable_backprop:						# step 18: 因为记录代数和标记变量的创造者都是为了反向传播，所以如果不需要反向传播，if里的文件就不需要使用
			self.generation = max([input.generation for input in inputs])
			for output in outputs:
				output.set_creator(self)					# 产生的数据保存创造者
		self.inputs = inputs							# 保存输入的数据 数据类型 tuple
		self.outputs = [weakref.ref(output) for output in outputs]	# step17 跟输出对象创建弱引用，确保输出对象对他来说无所谓，也不要被输出对象吊着口气
		return outputs if len(outputs) > 1 else outputs[0]
	
	def forward(self, xs):
		"""前向传播，要求子类自行实现，处理的数据类型为 np.ndarray
			function.__call__()中被调用，传入的数据类型是 np.ndarray，因为解包了

		Args:
			xs (np.ndarray): 函数接收的数据

		Raises:
			NotImplementedError: 如果子类未定义，则会自行报错
		"""
		raise NotImplementedError()
	
	def backward(self, gy):
		"""反向传播，要求子类自行实现，处理的数据类型为 np.ndarray
			在 Variable.backward()中被调用，传入的数据类型是 np.ndarray

		Args:
			gy (ndarray): 经过该函数的梯度

		Raises:
			NotImplementedError: _description_
		"""
		raise NotImplementedError()

class Square(Function):					# 在step13中需要修改的是 backward 中的 `self.input`,
	def forward(self, input_data):		# function被调用
		return input_data**2
	
	def backward(self, gy):				# var.backward中被调用
		x = self.inputs[0].data			# square本身为单输入，所以仅考虑第一个元素即可
		return 2*x*gy

class Exp(Function):
	def forward(self, input_data):
		return np.exp(input_data)
	
	def backward(self, gy):
		x = self.inputs[0].data
		return np.exp(x)*gy

class Add(Function):
	def forward(self, x0, x1):	# 修改以更符合常人阅读习惯
		y = x0 + x1
		return y

	def backward(self, gy):
		return gy, gy

class Mul(Function):
	def forward(self, x0, x1):
		y = x0 * x1
		return y
	
	def backward(self, gy):
		x0, x1 = self.inputs[0].data, self.inputs[1].data
		return x1 * gy, x0 * gy

class Neg(Function):
	def forward(self, xs):
		return -xs
	
	def backward(self, gy):
		return -gy

class Sub(Function):
	def forward(self, x0, x1):
		return x0 - x1
	
	def backward(self, gy):
		return gy, -gy

class Div(Function):
	def forward(self, x0, x1):
		return x0 / x1
	
	def backward(self, gy):
		x0, x1 = self.inputs[0].data, self.inputs[1].data
		gx0 = gy/x1
		gx1 = -gy * (x0 / (x1 ** 2))
		return gx0, gx1

class Pow(Function):
	def forward(self, x0, x1): 	# x0^x1
		return x0**x1
	
	def backward(self, gy):
		x0, x1 = self.inputs[0].data, self.inputs[1].data
		gx0 = x1 * (x0 ** (x1 - 1)) * gy
		gx1 = (x0**x1) * np.log(x0) * gy		#! 其实这里有个隐藏问题，对于 x0<0 log没意义，会nan。
		return gx0, gx1


def square(x):
	return Square()(x)

def exp(x):
	return Exp()(x)

def add(x0, x1):
	# x0 一般是调用add的var变量，x1可能是任何支持的变量
	x1 = as_array(x1)	# as_array 仅对标量有用，将变量转换为np.ndarry，否则直接返回
	return Add()(x0, x1)

def mul(x0, x1):
	x1 = as_array(x1)
	return Mul()(x0, x1)

def neg(x):	# neg 只需要考虑自身，不需要考虑其它类型变量的处理
	return Neg()(x)

def sub(x0, x1):
	x1 = as_array(x1)
	return Sub()(x0, x1)

def rsub(x0, x1):
	x1 = as_array(x1)
	return Sub()(x1, x0)

def div(x0, x1):
	x1 = as_array(x1)
	return Div()(x0, x1)

def rdiv(x0, x1):
	x1 = as_array(x1)
	return Div()(x1, x0)

def pow(x0, x1):
	x1 = as_array(x1)
	return Pow()(x0, x1)

def rpow(x0, x1):
	x1 = as_array(x1)
	return Pow()(x1, x0)

Variable.__add__ = add
Variable.__radd__ = add	# var在右侧时会调用 __radd__ 并将自己作为第一个参数
Variable.__mul__ = mul
Variable.__rmul__ = mul	# 同理
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__pow__ = pow
Variable.__rpow__ = rpow

###########################################
##				unittest test
###########################################
def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(x.data - eps)
	x1 = Variable(x.data + eps)
	y0 = f(x0)
	y1 = f(x1)
	return 	(y1.data - y0.data) / (2 * eps)

#? How to use it?
## python -m unittest .\step\stepxx.py 
class SquareTest(unittest.TestCase):
	def test_forward(self):
		x = Variable(np.array(2.0))
		y = square(x)
		expected = np.array(4.0)
		self.assertEqual(y.data, expected)
	def test_backward(self):
		x = Variable(np.array(3.0))
		y = square(x)
		y.backward()
		expected = np.array(6.0)
		self.assertEqual(x.grad, expected)
	def test_gradient(self):
		x = Variable(np.random.rand(3, 3))
		y = square(x)
		y.backward()
		num_grad = numerical_diff(square,x)
		flg = np.allclose(x.grad, num_grad)
		self.assertTrue(flg)

class ExpTest(unittest.TestCase):
	def test_forward(self):
		x = Variable(np.array(2.0))
		y = exp(x)
		expected = np.exp(2.0)
		self.assertEqual(y.data, expected)
	def test_backward(self):
		x = Variable(np.array(3.0))
		y = exp(x)
		y.backward()
		expected = np.exp(3)
		self.assertEqual(x.grad, expected)
	def test_gradient(self):
		x = Variable(np.random.rand(3, 3))
		y = exp(x)
		y.backward()
		num_grad = numerical_diff(exp,x)
		flg = np.allclose(x.grad, num_grad)
		self.assertTrue(flg)

class AddTest(unittest.TestCase):
	def test_forward(self):
		x0 = Variable(np.array(2))
		x1 = Variable(np.array(3))
		ys = add(x0,x1)
		print(ys.data)
		self.assertIsInstance(ys, Variable)
		self.assertEqual(ys.data, np.array(5))
		self.assertEqual(ys.creator.__class__, Add)
	def test_multi_2x(self):
		x0 = Variable(np.array(3))
		y0 = add(x0, x0)
		y0.backward()
		self.assertEqual(x0.grad, 2)
	def test_multi_3x(self):
		x0 = Variable(np.array(3))
		y0 = add(add(x0, x0), x0)
		y0.backward()
		self.assertEqual(x0.grad, 3)

class ComplexBackwardTest(unittest.TestCase):
	def test_complex_backward(self):
		x = Variable(np.array(2.0))
		a = square(x)
		y = add(square(a), square(a))
		y.backward()
		print(y.data)
		print(x.grad)
		self.assertEqual(x.grad, 2**6)

class Memorytest(unittest.TestCase):
	def test_ComplexNet(self):
		for i in range(10):
			# 这代码如果不能处理好 循环引用，内存直接得“爆炸”
			# 所以需要弱引用，这里的函数统一向后弱引用
			x = Variable(np.random.randn(10000))
			y = square(square(square(x)))

	def test_GradMemoory(self):
		x0 = Variable(np.array(3))
		x1 = Variable(np.array(6))
		t = add(x0, x1)
		y = add(x0, t)
		y.backward()
		print(y.grad, t.grad)
		print(x0.grad, x1.grad)

	def test_Config(self):
		Config.enable_backprop = True
		x = Variable(np.ones((100,100,100)))
		y = square(square(square(x)))
		
		Config.enable_backprop = False
		x = Variable(2*np.ones((100,100,100)))
		y = square(square(square(x)))
		print(y.data)
		Config.enable_backprop = True

	def test_WithConfig(self):
		Config.enable_backprop = True
		with no_grad():
			print(Config.enable_backprop)
			self.assertEqual(Config.enable_backprop, False)
			x = Variable(np.array([1,2,3,4]))
			y = square(exp(square(x)))
			print(y.data)
		print(Config.enable_backprop)
		self.assertEqual(Config.enable_backprop, True)

class Step19_test(unittest.TestCase):
	def test_1(self):
		x = Variable(np.array([[1,2,3,4], [5,6,7,8]]))
		y = Variable(None)
		print(x.size)
		print(x.shape)
		print(x.ndim)
		print(x.dtype)
		print(len(x))
		print(x)
		print(y)

class Step20_to_22_test(unittest.TestCase):
	def test_20_1(self):
		x0 = Variable(np.array(3))
		x1 = Variable(np.array(2))
		x2 = Variable(np.array(1))
		y = x0 * x1 + x2
		y.backward()
		print(y)
		print(x0.grad, x1.grad, x2.grad)
	def test_21_1(self):
		x0 = Variable(np.array(2.0))
		y = x0 + np.array(3.0)
		y.backward()
		print(y)
		print(x0.grad)
	def test_21_2(self):
		x0 = Variable(np.array(2.0))
		y = np.array(7) + 3 * x0 + 5
		y.backward()
		print(y)
		print(x0.grad)
	def test_22_1(self):
		print("test_22_1")
		x = Variable(np.array(2.0))
		y = -x
		y.backward()
		print(y.data, x.grad)
		x.clear_grad()
		y = 2.0 - x
		y.backward()
		print(y.data, x.grad)
		x.clear_grad()
		y = x - 1.0
		y.backward()
		print(y.data, x.grad)
	def test_22_2(self):
		print("test_22_2")
		x = Variable(np.array(2.0))
		y = 2.0 / x
		y.backward()
		print(y.data, x.grad)
		x.clear_grad()
		y = x / 3.0
		y.backward()
		print(y.data, x.grad)
		x.clear_grad()
	def test_22_3(self):
		print("test_22_3")
		x = Variable(np.array(2.0))
		c = 2.0
		y = x**c
		y.backward()
		print(y.data, x.grad)
		x.clear_grad()
		y = c**x
		y.backward()
		print(y.data, x.grad)
	def test_22_4(self):
		print("test_22_4")
		x0 = Variable(np.array(2.0))
		x1 = Variable(np.array(3.0))
		y = x0**x1
		y.backward()
		print(y.data, x0.grad, x1.grad)

if __name__ == '__main__':
    unittest.main()