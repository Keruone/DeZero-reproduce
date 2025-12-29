if '__file__' in globals():
	import os, sys
	sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import time
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

model = MLP([hidden_size, hidden_size, 10], activation=F.relu)	# 在 models.MLP 中修改了前向传播的内容，以测试dropout
optimizer = optimizers.SGD().setup(model)

if dezero.cuda.gpu_enable:
	train_loader.to_gpu()
	model.to_gpu()
	print("Training in GPU...")
else:
	print("Training in CPU...")

for epoch in range(max_epoch):
	start = time.time()
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

	elapsed_time = time.time() - start
	print('epoch: {}, elapsed_time: {:.2f}'.format(epoch + 1, elapsed_time))
	print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

with dezero.test_mode():
	start = time.time()
	sum_loss, sum_acc = 0, 0
	for x, t in test_loader:
		y = model(x)
		loss = F.softmax_cross_entropy(y, t)
		acc = F.accuracy(y, t)
		sum_loss += float(loss.data) * len(t)
		sum_acc += float(acc.data) * len(t)
	elapsed_time = time.time() - start
	print('\nTest Mode\nelapsed_time: {:.2f}'.format(elapsed_time))
	print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))

