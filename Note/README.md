# DeZero 学习记录
本文用来记录 复现DeZero 过程中，各种看书第一眼不能理解的、或书中略过的知识。

- [DeZero 学习记录](#dezero-学习记录)
	- [Step 12: `*`在python的部分作用](#step-12-在python的部分作用)
	- [step 14: `x.grad += gx` 会导致梯度出问题](#step-14-xgrad--gx-会导致梯度出问题)
		- [现回顾案发现场：](#现回顾案发现场)
		- [案件分析：](#案件分析)
		- [总结](#总结)
	- [step 15-17: 核心设计：职责分离](#step-15-17-核心设计职责分离)
		- [1. 为什么要做职责分离？](#1-为什么要做职责分离)
		- [2. 引入弱链接](#2-引入弱链接)
	- [step 18: yield与装饰器](#step-18-yield与装饰器)
		- [1. yield](#1-yield)
		- [2. @contextlib.contextmanager 装饰器](#2-contextlibcontextmanager-装饰器)
	- [Step 39: 书中未讲明的 utils.reshape\_sum\_backward() 函数分析](#step-39-书中未讲明的-utilsreshape_sum_backward-函数分析)
		- [1. 使用的场景](#1-使用的场景)
		- [2. 函数的具体分析](#2-函数的具体分析)
	- [Step 40: 书中未讲明的 utils.sum\_to 函数分析](#step-40-书中未讲明的-utilssum_to-函数分析)
		- [1. 你应该知道的 numpy 广播特点](#1-你应该知道的-numpy-广播特点)
		- [2. 具体函数分析](#2-具体函数分析)
	- [Step 44: `super()` 以及 `__setattr__`](#step-44-super-以及-__setattr__)
		- [1. `super()`](#1-super)
		- [2. `__setattr__` 即其赋值语法](#2-__setattr__-即其赋值语法)


---
## Step 12: `*`在python的部分作用
关于 "*" 在 python 的作用，可以尝试运行我在这里的[代码](../example/python_args_unpack_demo.py)

总结如下：
1. `*args` 作用：把「多个独立参数」打包成「元组」（永远是元组）",
2. `*列表` 作用：把「列表/元组」解包成「多个独立参数」",
3. **直接**传列表 → `args` = (列表,)（元组里只有1个元素：列表）",
4. **解包**传列表 → `args` = (列表元素1, 列表元素2, ...)（元组长度=列表长度）",
5. 易错点：直接传列表时，`args[0]` 是列表**本身**，不是列表里的元素"

---
## step 14: `x.grad += gx` 会导致梯度出问题
这个问题是在执行[step 14](../step/step14.py)时，出现的问题。
你应该可以在我的[代码](../step/step14.py)中发现有一块 `#!` 注释的地方，那里是案发现场

### 现回顾案发现场：
- 当时，代码为`x.grad += gx`。此时运行测试例程（如下）你会发现，结果是**4**（预期是 3）。对于像我一样底子是`C`打出来的，python是边用边学的，肯定此时会感到困惑！
	```python
		def test_multi_3x(self):
			x0 = Variable(np.array(3))
			y0 = add(add(x0, x0), x0)
			y0.backward()
			self.assertEqual(x0.grad, 3)
	```
- 然后展开调试，你会发现代码中的`gx`在第二次反向传播后仍然正确，但是如果你仔细观察`gy`，会发现怎么变成**2**了？？？😕

### 案件分析：
- 案发原因1：
  - 这里最抽象的地方，就是：python的赋值，其实是对象引用（也因为这个特性，python有什么深复制、浅复制等等奇怪东西）
  - 举个例子
  	```python
  	# 第一行：创建数组并赋值（对应你 x0.grad = gx）
  	a = np.array(1)
  	b = a
  	# 第二行：修改b，a也变了（对应你 gy 莫名变2）
  	b += 1; print(a, b)  # 输出：2 2（不是你以为的 1 2！）
  	```
  - 而这正式案发的**原因其一**！其实你仔细观察前面代码，可以发现全部的“赋值”都是引用！
	> ***\*注意***，Python传参是**传对象引用（pass-by-object-reference）**）
  - 所以在`add()`反向传播时，直接将`gys`返回了，然后又“赋值”到`gxs`，而`gx`又是直接从`gxs`中取出，又赋值给`x.grad`。那这么兜了一圈后，**`x.grad`和`gy`就给绑上了同一个对象**！（真是一对苦命鸳鸯啊😭）
- 案发原因2：
  - 在知道“对象引用”的特性后，你应该还要知道：大多数运算（如 a + b）会创建一个新对象，然后赋值操作（x = ...）会让变量名重新绑定到这个新对象。
  - 但是 `+=` 不是啊！
	> 原地运算符（如 +=, *=, etc.）对可变对象（如 list, np.ndarray）会尝试直接修改原对象的内容（in-place），而不改变其身份（id 不变）；对不可变对象（如 int, str），则退化为普通赋值（创建新对象）。
  - 这就破案了啊！逆天的原地运算会导致你的`gy`随着`x.grad`同步增加！

### 总结
这是 Python 新手（尤其 C 背景）极易踩的坑💔：既容易忽略「赋值 = 引用」的特性，又没意识到「原地运算符」对可变对象的修改会影响所有引用该对象的变量。

---
## step 15-17: 核心设计：职责分离
在实现 `generation` 赋值逻辑时，核心设计决策是：将「Variable 辈分设置」的逻辑放在 `Variable.set_creator` 中。
但是这让我想到了一个问题：为什么**选择`Variable.set_creator`而非 Function.__call__ 里实现**？
后来简单了解以下，得知可能与**职责分离**原则有关

### 1. 为什么要做职责分离？
- 反面设计（不可取）：Function 操控 Variable 内部属性
如果在 Function.__call__ 中直接修改 Variable 的 generation：
	```python
	# ❌ 违背职责分离：Function 侵入 Variable 内部逻辑
	def __call__(self, *inputs):
		# ... 其他逻辑 ...
		for output in outputs:
			output.creator = self
			output.generation = self.generation + 1  # Function 直接改 Variable 属性
	```
	问题：
	- 耦合度极高：Function 必须知道 Variable 有 generation 属性，且知道其计算规则（+1）；
	- 扩展性差：若后续修改 generation 计算规则（比如 +2），所有 Function 都要同步修改；
	- 违背「对象自治」：Variable 无法自己掌控核心属性，沦为 Function 的 “附属品”。
- 正面设计（代码中采用）：Variable 自治核心属性
	```python
	# ✅ 符合职责分离：Variable 自己管理核心属性
	class Variable:
		def set_creator(self, func):
			self.creator = func
			self.generation = func.generation + 1  # Variable 自主决定辈分计算规则
	```
	优势：
	- 边界清晰：
		- Function 仅负责「计算逻辑」+「通知 Variable：我创建了你」（调用 set_creator），无需知道 Variable 内部如何处理；
		- Variable 仅负责「管理自身数据 / 梯度 / 辈分」，无需依赖 Function 的实现细节；
	- 可维护性强：修改 generation 规则时，只改 Variable.set_creator 一个地方即可；
	- 符合直觉：就像 “孩子知道自己的父母（creator），并自主计算自己的辈分，而非父母强行贴标签”。

### 2. 引入弱链接
引入弱链接的缘由，除了书中提及的内存管理的原因（这当然也非常重要），我想某种角度来说也是跟**职责分离**相关的。
- 这里的弱链接主要修改对象还是函数，它确保了对于函数而言，它对调用它的对象（输入）保持强链接，对于它的计算结果（输出）保持了弱链接。
  > Function 只完成「生成输出」的核心职责，输出变量的生命周期由「使用它的代码」决定，而非生产者（Function）—— 彻底切断 Function 对输出的 “过度责任”。
- 而对于变量，它仅记住自己的创建者（输出自己的函数），但从不考虑谁会使用它。
  > 若 Variable 记录 “使用者”，会导致：① 耦合所有使用它的 Function/Variable；② 引入新的循环引用；③ 违背 “只关注自身来源” 的职责。
- 其实这也是一种遵循了职责分离的代码编写方式
- 最终实现：每个对象只对「自己职责范围内的依赖」负责，不插手「无关的对象生命周期」。

---
## step 18: yield与装饰器
> ？不要忘记，这份笔记记录的是书本上简略提及或者跳过的东西，所以看到标题和章节标题不符不要奇怪

### 1. yield
可以参考这篇[博客](https://blog.csdn.net/mieleizhi0522/article/details/82142856)
对于我们的代码来说，核心用到的特性，就是：
- 你第一次执行含有yield的函数时，程序会运行到yield的位置，然后表面上类似return一样返回
- 但是当你下一次调用该函数时，会从yield的下一行开始接着执行程序
- 在函数暂停（yield）时，所有属于该函数的局部变量、执行位置都会被栈保存，不会像return那样销毁

### 2. @contextlib.contextmanager 装饰器
1. 什么是装饰器？
	装饰器，装饰器，顾名思义，我们可以先暂时理解为：一个 “包装函数”，能在不改动原函数代码的前提下，给原函数加前置 / 后置逻辑。案例如下：
	```python
	# 定义装饰器：给函数加“执行日志”功能
	def log_decorator(func):
		# 包装函数：接收原函数的参数，执行增强逻辑
		def wrapper(*args, **kwargs):
			print(f"开始执行函数：{func.__name__}")  # 前置逻辑
			result = func(*args, **kwargs)          # 执行原函数
			print(f"函数执行完毕：{func.__name__}")  # 后置逻辑
			return result
		return wrapper

	# 使用装饰器：@+装饰器名，放在函数定义上方
	@log_decorator
	def square(x):
		return x * x

	# 调用原函数，自动触发装饰器逻辑
	print(square(2))
	# 输出：
	# 开始执行函数：square
	# 函数执行完毕：square
	# 4
	``` 
	你猜猜这个和什么等价？
	```python
	# 定义装饰器：给函数加“执行日志”功能
	def log_decorator(func):
		# 包装函数：接收原函数的参数，执行增强逻辑
		def wrapper(*args, **kwargs):
			print(f"开始执行函数：{func.__name__}")  # 前置逻辑
			result = func(*args, **kwargs)          # 执行原函数
			print(f"函数执行完毕：{func.__name__}")  # 后置逻辑
			return result
		return wrapper

	# 使用装饰器：@+装饰器名，放在函数定义上方
	def square(x):
		return x * x

	square = log_decorator(square)
	# 调用原函数，自动触发装饰器逻辑
	print(square(2))
	# 输出：
	# 开始执行函数：square
	# 函数执行完毕：square
	# 4
	``` 
	欸，看来是不是有点眉目了？说白了装饰器，说白了装饰器本质是返回“包装函数wrapper”的函数，wrapper里会中途调用原函数。然后为了偷懒，给你整了一个很抽象的写法 `@xxx`，并且要求你放在函数的正上方。额~不好评价。¯\\_(ツ)_/¯
	> 而这个很抽象的简写，就是所谓的 “语法糖”。🙃

2. 什么是 `with`
	with 是 Python 对「上下文管理器」的语法糖 —— 只要一个对象实现了 `__enter__` 和 `__exit__` 方法（也就是 “上下文管理器”），就能用 with 调用。
	- 进入 `with`
	- 执行 `__enter__`
	- 执行 `with` 的块内逻辑
	- 退出 `with`
	- 执行 `__exit__`
3. 什么是 `@contextlib.contextmanager`
   在你理解什么是装饰器后，看这个就容易理解多了：
   - 它接收一个「带 yield 的函数」，返回一个「包装后的上下文管理器函数」；
   - 这个装饰器是 “桥梁”，把一个带 yield 的普通函数，转换成能被 with 调用的上下文管理器：
     - 通过 yield 分割 “进入上下文” 和 “退出上下文” 的逻辑。
   - 用 with 调用时，装饰器生成的wrapper会自动处理：执行yield前逻辑 → 暂停（执行with块内代码）→ 执行yield后逻辑

---
## Step 39: 书中未讲明的 utils.reshape_sum_backward() 函数分析
### 1. 使用的场景
首先，我们先来看看函数调用的地方和使用的背景：
```python
class Sum(Function):
	def __init__(self, axis, keepdims):
		self.axis = axis
		self.keepdims = keepdims
	
	def forward(self, x):
		self.x_shape = x.shape
		return x.sum(axis = self.axis, keepdims = self.keepdims)	# 这里其实是 np.sum()
	
	def backward(self, gy):
		gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims) 	# 因为使用 axis 和 keepdims 会出现改变梯度形状的情况，所以要修正
		return broadcast_to(gy, self.x_shape)

def sum(x, axis = None, keepdims = False):
	return Sum(axis, keepdims)(x)
```
关于使用的原因，书中的原文如下。一言以蔽之，即 ***形状前后有变化，需要调整***。
> 在反向传播的实现中，我们在 broadcast_to 函数之前使用了 utils.reshape_sum_backward 函数。这个函数会对 gy 的形状稍加调整（因为使用 axis 和 keepdims 求和时会出现改变梯队形状的情况）。

### 2. 函数的具体分析
调用的函数代码如下
```python
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
```
- 变量初始化与轴的标准化
	```python
	ndim = len(x_shape)  				# 获取原输入x的维度数（比如x_shape=(2,3)，ndim=2）
	tupled_axis = axis
	if axis is None:
		tupled_axis = None 				# axis=None表示对所有元素求和（比如(2,3)→()）
	elif not isinstance(axis, tuple):
		tupled_axis = (axis,)  			# 把单个轴（如axis=1）转成元组(1,)，统一处理格式
	```
	- 目的：把各种形式的 axis（None / 单个 int / 元组）统一成 “None 或元组” 的格式，方便后续处理。
	- 举例：
		- 输入 axis=0 → 转成 (0,)
		- 输入 axis=(0,1) → 保持不变
		- 输入 axis=None → 保持 None
- 核心逻辑：判断是否需要插入维度
	```python
	if not (ndim == 0 or tupled_axis is None or keepdims):
		# 情况1：需要插入维度（最常见的场景）
		actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
		shape = list(gy.shape)			# 转换为列表
		for a in sorted(actual_axis):
			shape.insert(a, 1)			
			# insert 是 Python 列表（list）的内置方法 
			# 语法：`列表.insert(索引位置, 要插入的元素)`
			# 作用：在指定索引位置插入一个元素，原位置及之后的元素会自动向后移一位；
	else:
		# 情况2：不需要插入维度
		shape = gy.shape
	```
	- 判断条件 not (ndim == 0 or tupled_axis is None or keepdims)：
      - ndim == 0：原输入 x 是标量（shape=()），无需重塑。
      - tupled_axis is None：sum 时对所有维度求和（比如 x 是 (2,3)，sum 后是标量），gy 已是标量，无需重塑
      - keepdims=True：sum 时保留了维度（比如 (2,3) 沿 axis=1 sum 后是 (2,1)），gy 形状已匹配，无需插入 1
      - 只有以上都不满足时，才需要给 gy 插入维度 1
      - 即 **不是标量** 并且 **指定了axis** 并且 **没有开启keepdims**
    - if 框体解释
      - actual_axis：处理负轴（比如 axis=-1，ndim=2 → 实际是 axis=1），转成非负索引，避免越界。
      - shape.insert(a, 1)：在指定轴的位置插入 1，把 gy 的形状还原成 “压缩前的维度（只是压缩轴变成 1）”。
    - 作用：综上，将选定的求和导致被压缩掉的axis维度给他还原出来（在该维度尺寸设置为1即可），以便于广播
    - 举例：
      - 正向传播：x.shape=(2,3)，sum (axis=1) → 输出 shape=(2,)，keepdims=False
      - 反向传播：gy.shape=(2,)
        - actual_axis = [1]（axis=1 是正索引，无需转换）
        - shape 初始是 [2]
        - 对 sorted 后的 axis=1 执行 insert (1,1) → shape 变成 [2,1]
        - 最终 gy.reshape ([2,1])，后续广播就能还原成 (2,3)
- 执行重塑并返回
	```python
	gy = gy.reshape(shape)  # 把gy重塑成计算好的shape
	return gy
	```
	- 把 gy 的形状调整为插入了 1 的维度，为后续的广播（反向传播的梯度累加）做准备。

---
## Step 40: 书中未讲明的 utils.sum_to 函数分析
```python
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
```
### 1. 你应该知道的 numpy 广播特点
NumPy 的广播总是**从右向左对齐维度**，只在**左侧（前面）补充大小为 1 的维度**，且**仅当** **对应维度相等或其中一方为 1 时**才能广播，不能在右侧（末尾）新增维度。
- 正例：
	```python
	import numpy as np

	a = np.ones((3,))        # shape: (3,)
	b = np.ones((2, 4, 3))   # shape: (2, 4, 3)

	# a 会被广播为 (1, 1, 3) → 再广播为 (2, 4, 3)
	c = a + b
	print(c.shape)  # 输出: (2, 4, 3)
	```
	为什么合法？
    - NumPy 将 a.shape = (3,) 在左侧补 1 → (1, 1, 3)（符合“只在左侧补充维度”）；
    - 从右向左对齐：
      - 第 -1 维：3 vs 3 → 相等 ✅
      - 第 -2 维：1 vs 4 → 有 1 ✅
      - 第 -3 维：1 vs 2 → 有 1 ✅
      - 没有在末尾新增维度（a 的 3 在最右边，和 b 的 3 对齐）✅
- 负例1：
	```python
	import numpy as np

	a = np.ones((2, 3))      # shape: (2, 3)
	b = np.ones((2, 3, 4))   # shape: (2, 3, 4)

	# 尝试运算会报错！
	c = a + b  # ❌ ValueError!
	```
	为什么失败？
    - a.shape = (2, 3) 需要和 (2, 3, 4) 对齐；
    - NumPy 在左侧补 1：a → (1, 2, 3)；
    - 从右向左对齐比较：
      - 第 -1 维：3 vs 4 → 既不相等，也没有 1 ❌（后面的维度不用看了，已经失败）
- 负例2：
	```python
	import numpy as np

	a = np.ones((2, 4))      # shape: (2, 4)
	b = np.ones((2, 3, 4))   # shape: (2, 3, 4)

	# 尝试运算会报错！
	c = a + b  # ❌ ValueError!
	```
	为什么失败？
    - NumPy 将 a.shape = (2, 4) 在左侧补 1 → 变为 (1, 2, 4)
    - 从右向左对齐比较：
      - 第 -1 维：4 vs 4 → ✅ 相等
      - 第 -2 维：2 vs 3 → ❌ 既不相等，也没有 1
      - 第 -3 维：1 vs 2 → ✅ 有 1
    - 由于第 -2 维冲突，广播失败。

### 2. 具体函数分析
sum_to(x, shape) 的核心目标是：将通过广播扩展得到的数组 x，还原（压缩）回其原始形状 shape。这在反向传播中非常关键——当一个较小的张量参与前向计算并被广播成更大的张量时，其梯度会以更大形状返回；我们需要将这些梯度“聚合”回原始的小形状，而 sum_to 正是完成这一聚合操作的工具。

接下来函数逻辑逐行解析
```python
ndim = len(shape)
lead = x.ndim - ndim
```
- 关键假设：这些多出的维度一定在最前面（因为广播只在左侧补维）。*就是我前文叫你了解的内容嗷
- ndim：目标形状的维度数。
- lead：输入 x 比目标多出的维度数量。

```python
ndim = len(shape)
lead = x.ndim - ndim
```
- lead_axis = tuple(range(lead))

```python
axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])	# sx 是 sum_to 函数列表推导式中的临时变量，代表目标形状第 i 轴的尺寸值
```
- 遍历目标 shape，找出所有值为 1 的维度位置 i；
- 将其映射回 x 中的实际轴索引：i + lead（因为前面多了 lead 个维度）；
- 这些轴在前向传播中原本是 1，被广播成了更大的值（如 1 → 5），因此在反向传播中需要沿这些轴求和，把梯度“收回来”。
- 拆解成循环可能更好理解
	```python
	axis = []
	# 遍历目标形状 shape，i 是轴索引，sx 是该轴的尺寸值
	for i, sx in enumerate(shape):
		if sx == 1:  # 只关注目标形状中尺寸为1的轴
			axis.append(i + lead)
	axis = tuple(axis)
	```

```python
y = x.sum(lead_axis + axis, keepdims=True)
```
- 同时对 前导维度 和 原为 1 的维度 求和；
- keepdims=True 保证求和后这些维度仍保留为大小 1，便于后续形状调整。

```python
if lead > 0:
    y = y.squeeze(lead_axis)
```
- 删除前导维度（它们已经是大小为 1），使最终形状严格等于 shape。

---
## Step 44: `super()` 以及 `__setattr__`

### 1. `super()`
`super()`是python调用父类的某一个函数的办法。如你在子类和父类都同时定义了 foo 函数，但是用法略有不同，你在子类中希望实现新功能的同时，也有父类的功能，你有不像整个重写一遍父类的功能。这个时候，你就可以在子类中调用父类的foo来实现。
```python
class Parent:
    def foo(self, x):
        print("Parent foo:", x)

class Child(Parent):
    def foo(self, x, y):
        super().foo(x)  # 调用父类的 foo
        print("Child extra:", y)
```
不过注意：即使你的类看起来没有显式继承任何类（如 class MyClass:），它实际上仍然隐式继承自 object。因此，当你在自定义类中写 `super().__setattr__(name, value)` 时，最终会调用 object 类的默认 `__setattr__` 方法。

### 2. `__setattr__` 即其赋值语法
当你看完书中的代码时，可能又会对以下的内容产生困惑：
```python
layer = Layer()
layer.p1 = Parameter(np.array(1))
layer.p2 = Parameter(np.array(1))
```
你可能会疑惑：这是在做什么？
- 这是在**动态地为实例添加属性**（注意：是实例，不是类！）。
- 每次执行 `obj.attr = value`，Python 都会自动调用 `obj.__setattr__('attr', value)`。
> 子类的效果书中已经讲的比较清楚了，我这里就简单讲讲父类的[`object`类]的 `__setattr__` 方法）
- 语法：实例.属性名=属性值
- 效果：给当前**实例**（*注意，是实例，不是类！*）添加一个新的`属性`，该`属性`的名称记作 `传入的参数name`，该属性的数值赋值为`传入的参数value`
> ⚠️ 小心陷阱：在自定义 `__setattr__` 方法内部，**不要直接写** self.attr = value，否则会再次触发 `__setattr__`，导致无限递归！正确做法是使用 `super().__setattr__(name, value)` 或直接操作 `self.__dict__`。
> 
> *`self.__dict__` 是 Python 中每个对象实例（instance）自带的一个字典（dictionary）属性，用于存储该实例的所有可变属性（instance attributes）。*

---