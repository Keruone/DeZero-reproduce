# DeZero-reproduce
Refer to Saito Koji's "Building Deep Learning Framework", and reproduce the DeZero component within it.
参考斋藤康毅的《深度学习入门：自制框架》，并复现其中的DeZero。

---
本仓库除了一步一步复刻了 DeZero，也总结了一些书中没有提及或略过，但是还算是比较重要的内容。详见[这里](./Note/README.md)
> # 我的工作环境 版本出了一些冲突(nvcc 版本和 cuda 版本 和 cupy 版本) 不太好解决，就将[此处](./dezero/cuda.py)`的gpu_enable = False` 了

- 至此 2025.12.19 22:35:06 完成了全书的阅览，并完成了 Step1 - Step57 的全部内容。
- 由于 Step58-Step60 绝大多数内容 与其说是 "自制框架", 到更像 系列第二本 《深度学习进阶：自然语言处理》（该书后半部分就是 RNN、LSTM）
- 也正因此，Step58-Step60 的内容我只进行了快速阅览，未进行代码的实现。