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
	def __call__(self, *inputs):						## 修改为可接收多项 Var 输入，且不要求列表（若超过1个元素，会自动打包为list）
		xs = [input.data for input in inputs]
		ys = self.forward(*xs)							# step12: 解包，让多个x可以与具体调用的有多个参数的函数相匹配，也是这一步，让exp square可以正常运行
		if not isinstance(ys, tuple):					# step12: 如果不是元组
			ys = (ys,)									# 			则变为元组，确保可迭代
		outputs = [Variable(as_array(y)) for y in ys]	# 调用as_array,确保创建的是var 参数是np.ndarry
		for output in outputs:
			output.set_creator(self)					# 产生的数据保存创造者
		self.inputs = inputs							# 保存输入的数据，辅助反向传播
		self.outputs = outputs
		return outputs if len(outputs) > 1 else outputs[0]
	
	def forward(self, xs):
		"""前向传播，要求子类自行实现，处理的数据类型为 np.addary

		Args:
			xs (np.addary): 函数接收的数据

		Raises:
			NotImplementedError: 如果子类未定义，则会自行报错
		"""
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

class Add(Function):
	def forward(self, x0, x1):	# 修改以更符合常人阅读习惯
		y = x0 + x1
		return y

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

class AllTest(unittest.TestCase):
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
	def test_add(self):
		x0 = Variable(np.array(2))
		x1 = Variable(np.array(3))
		ys = add(x0,x1)
		print(ys.data)
		self.assertIsInstance(ys, Variable)
		self.assertEqual(ys.data, np.array(5))
		self.assertEqual(ys.creator.__class__, Add)

if __name__ == '__main__':
    unittest.main()