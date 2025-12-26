if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(3))
y = 6 - 100 / (1 + 3 * (-x) * 6 + 5) / 2 - 5
y.backward()

print(y.data, x.grad)
