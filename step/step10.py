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
	
	def set_creator(self, func):	
		"""用于指定父级，即是哪个函数计算得到的它，用于调用反向传播"""
		self.creator = func
	
	def backward(self):
		"使用循环的方式实现 backward"
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		funcs = [self.creator]
		while funcs:
			f = funcs.pop()
			input_var, output_var = f.input, f.output
			input_var.grad = f.backward(output_var.grad)
			if input_var.creator is not None:
				funcs.append(input_var.creator)

def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(as_array(y))	# 调用as_array,确保
		output.set_creator(self)			# 产生的数据保存创造者
		self.input = input				# 保存输入的数据，辅助反向传播
		self.output = output
		return output
	
	def forward(self, input_data):		# 要求子类重新实现
		raise NotImplementedError()
	
	def backward(self, gy):
		raise NotImplementedError()

class Square(Function):					# 继承 Function
	def forward(self, input_data):		# 子类重定义forward函数
		return input_data**2
	
	def backward(self, gy):
		x = self.input.data
		return 2*x*gy

class Exp(Function):
	def forward(self, input_data):
		return np.exp(input_data)
	
	def backward(self, gy):
		x = self.input.data
		return np.exp(x)*gy

def square(x):
	return Square()(x)

def exp(x):
	return Exp()(x)

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