import os
import subprocess

def _dot_var(v, verbose = False):
	dot_var = '{} [label="{}", color=orange, style=filled]\n'
	name = '' if v.name is None else v.name
	if verbose and v.data is not None:
		if v.name is not None:
			name += ': '
		name += str(v.shape) + ' ' + str(v.dtype)
	return dot_var.format(id(v), name)

def _dot_func(f):
	dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
	txt = dot_func.format(id(f), f.__class__.__name__)
	
	dot_edge = "{} -> {}\n"
	for x in f.inputs:
		txt +=dot_edge.format(id(x), id(f))
	for y in f.outputs:
		txt +=dot_edge.format(id(f), id(y()))	# 输出是弱引用 weakref
	
	return txt

def get_dot_graph(output, verbose=False):
	txt = ''
	funcs = []
	seen_set = set()
	def add_func(f):
		if f not in seen_set:
			funcs.append(f)
			seen_set.add(f)
	
	add_func(output.creator)
	txt += _dot_var(output, verbose)

	while funcs:
		f = funcs.pop()
		txt += _dot_func(f)
		for x in f.inputs:
			txt += _dot_var(x, verbose)
			if x.creator is not None:
				add_func(x.creator)
	
	return 'digraph g{\n ' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
	dot_graph = get_dot_graph(output, verbose)

	# 1. 保存dot数据至文件
	tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
	if not os.path.exists(tmp_dir):
		os.mkdir(tmp_dir)
	graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
	print('Save to: ' + graph_path)

	with open(graph_path, 'w') as f:
		f.write(dot_graph)
	
	# 2. 调用dot命令
	extension = os.path.splitext(to_file)[1][1:]
	cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
	subprocess.run(cmd, shell=True)

def sum_to(x, shape):
    """Sum elements along axes to output an array of a given shape.

    Args:
        x (ndarray): Input array.
        shape:

    Returns:
        ndarray: Output array of the shape.
    """
	#! NumPy 的广播总是从右向左对齐维度，只在左侧（前面）补充大小为 1 的维度，且仅当对应维度相等或其中一方为 1 时才能广播，不能在右侧（末尾）新增维度。
    ndim = len(shape)
    lead = x.ndim - ndim
	# lead = x.ndim - ndim 只能表示“前面多出来的维度”，不能处理“后面多出来的维度”。这是因为 NumPy 的广播规则是从后往前对齐维度的，而该函数的设计正是基于这一规则。

    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)	# 只向压缩掉的维度和为1的维度进行sum求和
    if lead > 0:
        y = y.squeeze(lead_axis)				# 将lead的维度删除掉，比如 shape(1,2,3)->(squeeze)->(2,3)
    return y

def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """Reshape gradient appropriately for dezero.functions.sum's backward.

    Args:
        gy (dezero.Variable): Gradient variable from the output by backprop.
        x_shape (tuple): Shape used at sum function's forward.
        axis (None or int or tuple of ints): Axis used at sum function's
            forward.
        keepdims (bool): Keepdims used at sum function's forward.

    Returns:
        dezero.Variable: Gradient variable which is reshaped appropriately
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy