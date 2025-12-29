from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L

class Model(Layer):
	def plot(self, *inputs, to_file = 'model.png'):
		y = self.forward(*inputs)
		utils.plot_dot_graph(y, verbose=True, to_file=to_file)

class MLP(Model):
	def __init__(self, fc_connect_sizes, activation = F.sigmoid):
		super().__init__()
		self.activation = activation
		self.layers = []

		for i, out_size in enumerate(fc_connect_sizes):
			layer = L.Linear(out_size)
			setattr(self, 'l' + str(i), layer)
			self.layers.append(layer)

	def forward(self, x):
		for l in self.layers[:-1]:
			x = self.activation(l(x))
			# x = self.activation(F.dropout(l(x)))	# 这里dropout 加入只是验证 Step54.py
		return self.layers[-1](x)