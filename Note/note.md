# DeZero 学习记录
本文用来记录 复现DeZero 过程中，各种看书第一眼不能理解的、或书中略过的知识。

[toc]

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