import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		output = Variable(y)
		return output
	
	def forward(self, input_data):	# 要求子类重新实现
		raise NotImplementedError()

class Square(Function):				# 继承 Function
	def forward(self, input_data):	# 子类重定义forward函数
		return input_data**2

class Exp(Function):
	def forward(self, input_data):
		return np.exp(input_data)

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
	x = Variable(np.array([0.5,3,4]))
	dy = numerical_diff(f,x)
	print(dy)