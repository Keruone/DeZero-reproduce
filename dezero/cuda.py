import numpy as np
gpu_enable = False	# 我的电脑 版本出了一些冲突(nvcc 版本和 cuda 版本 和 cupy 版本) 不太好解决，就False了
try:
	import cupy as cp
	cupy = cp
except ImportError:
	gpu_enable = False
from dezero import Variable

def get_array_module(x):
	if isinstance(x, Variable):
		x = x.data
	if not gpu_enable:
		return np
	
	xp = cp.get_array_module(x)
	return xp

def as_numpy(x):
	if isinstance(x, Variable):
		x = x.data
	if np.isscalar(x):
		return np.array(x)
	elif isinstance(x, np.ndarray):
		return x
	return cp.asnumpy(x)

def as_cupy(x):
	if isinstance(x, Variable):
		x = x.data
	if not gpu_enable:
		raise Exception('Cupy cannot be loaded. Install Cupy!')
	return cp.asarray(x)
