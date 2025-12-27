if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(2))
y = F.log(x, base = 10)
y.backward(create_graph=True)
iters = 5
for i in range(iters):
	gx = x.grad
	x.clear_grad()
	gx.backward(create_graph=True)
	print(x.grad)