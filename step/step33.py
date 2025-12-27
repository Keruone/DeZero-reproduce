if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

# def f(x):
# 	y = x ** 4 - 2 * x ** 2
# 	return y
# x = Variable(np.array(2))
# y = f(x)
# y.backward(create_graph=True)
# print(x.grad)
# print(y)
# gx = x.grad
# x.clear_grad()
# gx.backward()
# print(x.grad)


# x = Variable(np.array(2))
# y = Variable.log(x, base = 10)
# print(y)
# y.backward(create_graph = True)
# print(x.grad)
# gx = x.grad
# x.clear_grad()
# gx.backward()
# print(x.grad)


def f(x):
	y = x ** 4 - 2 * x ** 2
	return y
x = Variable(np.array(2.0))		# 创建时一定得是 浮点数啊
iters = 10
for i in range(iters):
	print(i, x)
	y = f(x)
	x.clear_grad()
	y.backward(create_graph = True)
	gx = x.grad
	x.clear_grad()
	gx.backward()
	gx2 = x.grad
	x.data -= gx.data/gx2.data	# 不然像这里赋值的时候，右侧肯定时浮点数，左侧是整形，不让赋值的
	#! 原地赋值(in-place)很坑啊，只有原地赋值才会产生注释的问题