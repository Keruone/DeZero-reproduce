if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = dezero.DataLoader(train_set, batch_size)
test_loader = dezero.DataLoader(test_set, batch_size, shuffle=False)

model = MLP([hidden_size, hidden_size, 10], activation=F.relu)
optimizer = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
	sum_loss, sum_acc = 0, 0

	for x, t in train_loader:		# 因为不是 __call__ 所以不需要添加括号
		y = model(x)		# __call__ 里面额外的处理封装好了的，尽量不要使用 forward
		loss = F.softmax_cross_entropy(y, t)
		acc = F.accuracy(y, t)
		model.clear_grads()
		loss.backward()
		optimizer.update()

		sum_loss += float(loss.data) * len(t)
		sum_acc += float(acc.data) * len(t)

	print('epoch: {}'.format(epoch + 1))
	print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

	sum_loss, sum_acc = 0, 0
	with dezero.no_grad():
		for x, t in test_loader:	# 因为不是 __call__ 所以不需要添加括号
			y = model(x)	# __call__ 里面额外的处理封装好了的，尽量不要使用 forwardv
			loss = F.softmax_cross_entropy(y, t)
			acc = F.accuracy(y, t)
			sum_loss += float(loss.data) * len(t)
			sum_acc += float(acc.data) * len(t)
	print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))
