import numpy as np

class Variable:
	def __init__(self, data):
		self.data = data
		self.grad = None

class Function:
	def __call__(self, input):
		x = input.data
		y = self.forward(x)
		self.input = input	# 保存输入的数据，辅助反向传播
		output = Variable(y)
		return output
	
	def forward(self, input_data):	# 要求子类重新实现
		raise NotImplementedError()
	
	def backward(self, gy):
		raise NotImplementedError()

class Square(Function):				# 继承 Function
	def forward(self, input_data):	# 子类重定义forward函数
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

	x = Variable(np.array([0.5,2,3]))
	y = A(B(C(x)))
	dy = np.array(1)
	dx = C.backward(dy)
	dx = B.backward(dx)
	dx = A.backward(dx)
	print(dx)