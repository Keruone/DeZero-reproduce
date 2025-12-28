if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers
from dezero.models import MLP


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000
hidden_size = 10

model = MLP((hidden_size,1))
optimizer = optimizers.SGD(lr)
optimizer = optimizers.MomentumSGD(lr)
optimizer.setup(model)
for i in range(iters):
	y_pred = model(x)	# 有 __call__ 能调用
	loss = F.mean_squared_error(y, y_pred)

	model.clear_grads()
	loss.backward()

	optimizer.update()
	if i % 1000 == 0:
		print(loss)
