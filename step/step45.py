if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
import dezero.models as M


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000
hidden_size = 10

class TwoLayersNet(Model):
	def __init__(self, hidden_size, out_size):
		super().__init__()
		self.l1 = L.Linear(hidden_size)
		self.l2 = L.Linear(out_size)

	def forward(self, x):
		y = self.l1(x)
		y = F.sigmoid(y)
		y = self.l2(y)
		return y
	
	# clear_grad 已经再 基类 中实现了

model = TwoLayersNet(hidden_size, 1)

for i in range(iters):
	y_pred = model(x)	# 有 __call__ 能调用
	loss = F.mean_squared_error(y, y_pred)

	model.clear_grads()
	loss.backward()

	for p in model.params():
		p.data -= lr * p.grad.data
	if i % 1000 == 0:
		print(loss)

print("MLP test")
model = M.MLP((10,1))
for i in range(iters):
	y_pred = model(x)	# 有 __call__ 能调用
	loss = F.mean_squared_error(y, y_pred)

	model.clear_grads()
	loss.backward()

	for p in model.params():
		p.data -= lr * p.grad.data
	if i % 1000 == 0:
		print(loss)
