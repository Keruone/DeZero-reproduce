if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = x.reshape(2,3)
print(y)
y.backward()
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = F.transpose(x)
print(y)
y.backward()
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = x.transpose()
print(y)
y.backward()
print(x.grad)

x = Variable(np.array([[1,2,3],[4,5,6]]))
y = x.T
print(y)
y.backward()
print(x.grad)