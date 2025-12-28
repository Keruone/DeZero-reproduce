if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array([[1,2,3], [4,5,6]]))
W = Variable(np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]]))
y = F.matmul(x, W)
print(y)
y.backward()
print(x.grad.shape, W.grad.shape)
print(x.grad)
print(W.grad)