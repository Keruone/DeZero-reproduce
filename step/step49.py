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

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

transformer = lambda x: x / 2
train_set = dezero.datasets.Spiral(transformer = transformer)	# 仅做功能测试，不反应实际效果
model = MLP([hidden_size, 3])
optimizer = optimizers.SGD(lr = lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)	# 向下取整

for epoch in range(max_epoch):
	index = np.random.permutation(data_size)
	sum_loss = 0

	for i in range(max_iter):
		batch_index = index[i * batch_size:(i + 1) * batch_size]	# numpy 切片自己会末端 -1
		# batch = train_set[batch_index]
		batch = [train_set[index] for index in batch_index]
		batch_x = np.array([example[0] for example in batch])
		batch_t = np.array([example[1] for example in batch])
		y = model.forward(batch_x)
		loss = F.softmax_cross_entropy(y, batch_t)
		# print(loss)
		model.clear_grads()
		loss.backward()
		optimizer.update()
		sum_loss += float(loss.data) * len(batch_t)

	avg_loss = sum_loss / data_size
	print('epoch: %d, loss: %.2f' % (epoch + 1, avg_loss))	# % 是 Python 中的 字符串格式化运算符，作用是将括号里的变量值，按指定格式嵌入到前面的字符串中。

