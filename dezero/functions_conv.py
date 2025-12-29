import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize
from dezero.functions import linear, broadcast_to

def conv2d_simple(x, W, b=None, stride=1, pad=0):
	x, W = as_variable(x), as_variable(W)
	Weight = W
	
	N, C, H, W = x.shape
	OC, C, KH, KW = Weight.shape
	SH, SW = pair(stride)
	PH, PW = pair(pad)
	OH = get_conv_outsize(H, KH, SH, PH)
	OW = get_conv_outsize(W, KW, SW, PW)
	col = im2col(x, (KH, KW), stride, pad, to_matrix=True)
	Weight = Weight.reshape(OC, -1).transpose()			# 不要担心，function做过reshape的反向传播
	# 这里返回的 t 是 N*OH*OW 行，OC 列
	t = linear(col, Weight, b)
	# 从左到右遍历维度索引，先固定前面的维度，把最后一个维度填满，再往前推进一个维度。
	# 也因为上面 t 是 N*OH*OW 行 OC 列，所以这样reshape会去没有问题
	y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)	# 不要担心，function做过reshape的反向传播
	return y

def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)	# 每一行去一个
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)
    return y

# =============================================================================
#  conv2d / deconv2d
# =============================================================================
class Conv2d(Function):
	def __init__(self, stride=1, pad=0):
		super().__init__()
		self.stride = pair(stride)
		self.pad = pair(pad)
	def forward(self, x, W, b):
		xp = cuda.get_array_module(x)
		KH, KW = W.shape[2:]	# OC, C, KH, KW
		col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)
		y = xp.tensordot(col, W, ((1,2,3), (1,2,3)))
		if b is not None:
			y += b
		y = xp.rollaxis(y, 3, 1)
		# y = np.transpose(y, (0, 3, 1, 2))
		return y
	def backward(self, gy):
		x, W, b = self.inputs
		gx = deconv2d(gy, W, b=None, stride=self.stride, pad=self.pad, outsize = (x.shape[2], x.shape[3]))
		gW = Conv2DGradW(self)(x, gy)
		gb = None
		if b.data is not None:
			gb = gy.sum(axis=(0, 2, 3))
		return gx, gW, gb
def conv2d(x, W, b=None, stride=1, pad=0):
	return Conv2d(stride, pad)(x, W, b)

class Deconv2d(Function):
	def __init__(self, stride=1, pad=0, outsize=None):
		super().__init__()
		self.stride = pair(stride)
		self.pad = pair(pad)
		self.outsize = outsize
	def forward(self, x, W, b):
		xp = cuda.get_array_module(x)
		Weight = W
		SH, SW = self.stride
		PH, PW = self.pad
		C, OC, KH, KW = Weight.shape
		N, C, H, W = x.shape
		if self.outsize is None:
			out_h = get_deconv_outsize(H, KH, SH, PH)
			out_w = get_deconv_outsize(W, KW, SW, PW)
		else:
			out_h, out_w = pair(self.outsize)
		img_shape = (N, OC, out_h, out_w)
		gcol = xp.tensordot(Weight, x, (0, 1))
		gcol = xp.rollaxis(gcol, 3)
		y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
							to_matrix=False)
		# b, k, h, w
		if b is not None:
			self.no_bias = True
			y += b.reshape((1, b.size, 1, 1))
		return y
	def backward(self, gy):
		x, W, b = self.inputs
		gx = conv2d(gy, W, b=None, stride=self.stride, pad=self.pad)
		f = Conv2DGradW(self)
		gW = f(gy, x)
		gb = None
		if b.data is not None:
			gb = gy.sum(axis=(0, 2, 3))
		return gx, gW, gb
def deconv2d(x, W, b=None, stride=1, pad=0, outsize=None):
	return Deconv2d(stride, pad, outsize)(x, W, b)

class Conv2DGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        kh, kw = W.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = cuda.get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        gW = xp.tensordot(gy, col, ((0, 2, 3), (0, 4, 5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gW, stride=self.stride, pad=self.pad,
                      outsize=(xh, xw))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# =============================================================================
#  pooling(max-pooling) / average_pooling
# =============================================================================
class Pooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy):
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    def __init__(self, mpool2d):
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy):
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = (self.indexes.ravel()
                   + xp.arange(0, self.indexes.size * KH * KW, KH * KW))
        
        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride,
                          self.pad, to_matrix=False)
        return gx

    def backward(self, ggx):
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    def __init__(self, mpool2d):
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(x, kernel_size, stride=1, pad=0):
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    def __init__(self, kernel_size, stride=1, pad=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad,
                           to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy):
        # TODO(Koki): This is simple implementation
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= (KW*KH)
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N*C*OH*OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(gcol, self.input_shape, self.kernel_size, self.stride,
                    self.pad, to_matrix=False)
        return gx

# =============================================================================
#  im2col / col2im
# =============================================================================
def average_pooling(x, kernel_size, stride=1, pad=0):
    return AveragePooling(kernel_size, stride, pad)(x)


class Im2col(Function):
	def __init__(self, kernel_size, stride, pad, to_matrix):
		super().__init__()
		self.input_shape = None
		self.kernel_size = kernel_size
		self.stride = stride
		self.pad = pad
		self.to_matrix = to_matrix
	
	def forward(self, x):
		self.input_shape = x.shape
		y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
		return y
	
	def backward(self, gy):
		gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
		return gx
		
def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
	return Im2col(kernel_size, stride, pad, to_matrix)(x)

class Col2im(Function):
	def __init__(self, input_shape, kernel_size, stride, pad, to_matrix):
		super().__init__()
		self.input_shape = input_shape
		self.kernel_size = kernel_size
		self.stride = stride
		self.pad = pad
		self.to_matrix = to_matrix
	
	def forward(self, x):
		y = col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)
		return y
	
	def backward(self, gy):
		gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)
		return gx
def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
	return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)

def im2col_array(img, kernel_size, stride, pad, to_matrix):
	N, C, H, W = img.shape
	KH, KW = pair(kernel_size)
	SH, SW = pair(stride)
	PH, PW = pair(pad)
	OH = get_conv_outsize(H, KH, SH, PH)
	OW = get_conv_outsize(W, KW, SW, PW)

	xp = cuda.get_array_module(img)
	if xp!=np:
		col = _im2col_gpu(img, kernel_size, stride, pad)
	else:
		# np.pad 的第二个参数是依次 对每个维度的填充 (上填充, 下填充)
		# N, C 不填充 H, W 填充
		# 但为了正确还原，需要知道原始输入周围“可能被卷积核覆盖到的最大区域”。
		# 由于 stride > 1 时，最后一个窗口可能只覆盖部分区域，所以右边/下边需要额外填充 stride - 1。
		img = np.pad(img, ((0,0),(0,0),(PH, PH + SH -1),(PW, PW + SW - 1)))
		col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)
		# 不用 H, W 而用 KH, KW 的原因是可能stride不够(KH*SH<=H, KW*SW<=W)，最后有剩下的，所以只保留能被计算的部分
		
		# 循环提取每次卷积运算的对象
		for j in range(KH):			#* 注意这里是 kernel_H
			j_lim = j + SH * OH			# 即找到kernel的某个元素，在H方向上 于图像中最远可以到达的距离
			for i in range(KW):		#* 注意这里是 kernel_W
				i_lim = i + SW * OW		# 即找到kernel的某个元素，在W方向上 于图像中最远可以到达的距离
				# 因为前面是以 kernel 的 H,W 作为循环的参数的，所以这里slice的步长是SH和SW
				# SH步长恰好输出OH个，SW步长恰好输出OW个
				col[:,:,j,i,:,:] = img[:,:,j:j_lim:SH,i:i_lim:SW]
	if to_matrix:
		# 核心是改变排序以使得矩阵运算能够匹配
		# 改变后排序对应: N, OH, OW, C, KH, KW
		# 其中 (N, OH, OW) 为每次运算的一个单位，N个批次，每个批次均为 OH*OW个行
		# 其中 (C, KH, KW) 是矩阵中每一行应该有的具体展开内容，所以一起放在最后3轴
		col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N*OH*OW, -1)
	return col

def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
	N, C, H, W = img_shape
	KH, KW = pair(kernel_size)
	SH, SW = pair(stride)
	PH, PW = pair(pad)
	OH = get_conv_outsize(H, KH, SH, PH)
	OW = get_conv_outsize(W, KW, SW, PW)

	if to_matrix:	# im2col 的反向操作，先reshape，再重新给轴排序
		col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
	
	xp = cuda.get_array_module(col)
	if xp != np:
		img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
		return img
	else:
		# im2col 生成的矩阵肯定可以赋值的满，但是此处的不一定，所以是 zeros
		img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype)
		for j in range(KH):
			j_lim = j + SH * OH
			for i in range(KW):
				i_lim = i + SW * OW
				img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
		# 注意，此处的意思是 从 PH 到 H + PH ，不是从PH到H再奇奇怪怪的加一个PH
		return img[:, :, PH:(H+PH), PW:(W+PW) ]

#*==========================================
#*		copied from original code
#*==========================================
def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img