import numpy as np
class Optimizer:
	def __init__(self):
		self.target = None
		self.hook = []
	
	def setup(self, target):
		self.target = target
		return self	# 返回自身的引用
	
	def update(self):
		# 仅更新 梯度不是None， 即需要更新的参数
		params = [p for p in self.target.params() if p.grad is not None]

		# 预处理	#? 目前暂不清楚预处理的必要
		for f in self.hook:
			f(params)
		
		for param in params:
			self.update_one(param)
	
	def update_one(self, param):
		raise NotImplementedError()
	
	def add_hook(self, f):
		self.hook.append(f)

class SGD(Optimizer):
	def __init__(self, lr = 0.01):
		super().__init__()
		self.lr = lr
	
	def update_one(self, param):
		param.data -= param.grad.data * self.lr	# 因为目的是让损失尽量“小”， 所以沿梯度相反方向

class MomentumSGD(Optimizer):
	def __init__(self, lr = 0.01, momentum = 0.9):
		super().__init__()
		self.lr = lr
		self.momentum = momentum
		self.vs = {}
	
	def update_one(self, param):
		v_key = id(param)	# 以 id 作为 key，确保唯一性
		if v_key not in self.vs:
			self.vs[v_key] = np.zeros_like(param.data)
		v = self.vs[v_key]
		v *= self.momentum
		v -= self.lr * param.grad.data
		param.data += v