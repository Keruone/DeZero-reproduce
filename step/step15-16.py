import numpy as np
import unittest

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{} is not supported'.format(type(data))) 

		self.data = data
		self.grad = None
		self.creator = None
		self.generation = 0
	
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
		
	def backward(self):
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
			xs, ys = f.inputs, f.outputs
			gys = [y.grad for y in ys]
			gxs = f.backward(*gys)
			if not isinstance(gxs, tuple):
				gxs = (gxs,)
			for gx, x in zip(gxs, xs):	# xs与gxs一定是可以匹配上的
				if x.grad == None:
					x.grad = gx
				else:
					x.grad = x.grad + gx	#! 这里千万不能是 x.grad += gx
					#! 1. Python 中变量赋值本质上是“名字绑定”（name binding），即让一个名字（变量）指向某个对象 —— 可以理解为“引用”。
					#! 2. 大多数运算（如 a + b）会创建一个新对象，然后赋值操作（x = ...）会让变量名重新绑定到这个新对象。
					# !3. 原地运算符（如 +=, *=, etc.）对可变对象（如 list, np.ndarray）会尝试直接修改原对象的内容（in-place），而不改变其身份（id 不变）；对不可变对象（如 int, str），则退化为普通赋值（创建新对象）。
				if x.creator is not None:
					add_func(x.creator)


def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x

class Function:
	def __call__(self, *inputs):						## 修改为可接收多项 Var 输入，会打包为tuple
		self.generation = max([input.generation for input in inputs])
		xs = [input.data for input in inputs]
		ys = self.forward(*xs)							# step12: 解包，让多个x可以与具体调用的有多个参数的函数相匹配，也是这一步，让exp square可以正常运行
		if not isinstance(ys, tuple):					# step12: 如果不是元组
			ys = (ys,)									# 			则变为元组，确保可迭代
		outputs = [Variable(as_array(y)) for y in ys]	# 调用as_array,确保创建的是var 参数是np.ndarry
		for output in outputs:
			output.set_creator(self)					# 产生的数据保存创造者
		self.inputs = inputs							# 保存输入的数据 数据类型 tuple
		self.outputs = outputs							# 保存输出的数据 数据类型 tuple
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

def square(x):
	return Square()(x)

def exp(x):
	return Exp()(x)

def add(x0, x1):
	return Add()(x0, x1)

def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(x.data - eps)
	x1 = Variable(x.data + eps)
	y0 = f(x0)
	y1 = f(x1)
	return 	(y1.data - y0.data) / (2 * eps)

def f(x):
	A = Square()
	B = Exp()
	C = Square()
	return C(B(A(x))) 

###########################################
##				unittest test
###########################################
def numerical_diff(f, x, eps=1e-4):
	x0 = Variable(x.data - eps)
	x1 = Variable(x.data + eps)
	y0 = f(x0)
	y1 = f(x1)
	return 	(y1.data - y0.data) / (2 * eps)

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
	def test_backward(self):
		x0 = Variable(np.array(2))
		x1 = Variable(np.array(3))
		ys = add(x0,x1)
		ys.backward()
		print(ys.data)
		self.assertIsInstance(ys, Variable)
		self.assertEqual(ys.grad, x0.grad)
		self.assertEqual(ys.grad, x1.grad)
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


if __name__ == '__main__':
    unittest.main()