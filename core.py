import numpy as np

class Variable:
	def __init__(self, data):
		if data is not None:
			if not isinstance(data, np.ndarray):
				raise TypeError('{} is not supported'.format(type(data))) 

		self.data = data
		self.grad = None
		self.creator = None
	
	def set_cretor(self, func):	
		"""用于指定父级，即是哪个函数计算得到的它，用于调用反向传播"""
		self.creator = func
	
	def backward(self):
		"使用循环的方式实现 backward"
		if self.grad is None:
			self.grad = np.ones_like(self.data)

		func = [self.creator]
		while func:
			f = func.pop()
			input, output = f.input, f.output
			input.grad = f.backward(output.grad)
			if input.creator is not None:
				func.append(input.creator)

def as_array(x):
	if np.isscalar(x):
		return np.array(x)
	return x

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(as_array(y))	# 调用as_array,确保
		output.set_cretor(self)			# 产生的数据保存创造者
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

if __name__ == "__main__":
	A = Square()
	B = Exp()
	C = Square()

	x = Variable(np.array([0.5,1,3]))
	a = A(x)
	b = B(a)
	y = C(b)
	assert y.creator == C
	assert y.creator.input == b
	assert y.creator.input.creator == B
	assert y.creator.input.creator.input == a
	assert y.creator.input.creator.input.creator == A
	assert y.creator.input.creator.input.creator.input == x
	y.grad = np.array(1)
	y.backward()
	print(x.grad)