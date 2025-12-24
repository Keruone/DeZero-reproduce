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

if __name__ == "__main__":
	A = Square()
	B = Exp()
	C = Square()

	x = Variable(np.array([0.5,2,3]))
	y = C(B(A(x)))

	print(type(y))
	print(y.data)