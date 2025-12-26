# ===================== 基础概念：*args 的核心作用 =====================
print("===== 基础1：*args 是「打包」多个独立参数为元组 =====")
# 定义函数：*args 接收任意个数的位置参数
def test_pack(*args):
    print(f"args 的类型: {type(args)}")  # 永远是元组
    print(f"args 的值: {args}")
    print(f"args 的长度: {len(args)}")
    return args

# 场景1：不传参数 → args 是空元组
print("\n【场景1】不传参数")
test_pack()  # 输出：args=()

# 场景2：传1个独立参数 → args 是单元素元组
print("\n【场景2】传1个独立参数（整数）")
test_pack(10)  # 输出：args=(10,)

# 场景3：传多个独立参数 → args 是多元素元组
print("\n【场景3】传多个独立参数（整数+字符串）")
test_pack(10, "hello", True)  # 输出：args=(10, 'hello', True)

# ===================== 关键易错点：列表的传参方式 =====================
print("\n===== 基础2：列表的两种传参方式（新手必看） =====")
# 准备测试列表
my_list = [1, 2, 3]

# 场景4：直接传列表（作为1个独立参数）→ args 是「元组里包列表」
print("\n【场景4】直接传列表（未解包）")
test_pack(my_list)  # 输出：args=([1,2,3],) → 元组长度=1

# 场景5：解包传列表（加*）→ 列表元素变成多个独立参数
print("\n【场景5】解包传列表（加*）")
test_pack(*my_list)  # 输出：args=(1,2,3) → 元组长度=3

# ===================== 进阶：自定义对象的传参（贴近你的实际场景） =====================
print("\n===== 进阶：自定义对象（如Variable）的传参 =====")
# 模拟你之前的 Variable 类（简化版）
class Variable:
    def __init__(self, data):
        self.data = data
    # 自定义打印格式，方便看结果
    def __repr__(self):
        return f"Variable({self.data})"

# 准备 Variable 实例
x0 = Variable(2)
x1 = Variable(3)
var_list = [x0, x1]

# 场景6：直接传 Variable 列表（未解包）
print("\n【场景6】直接传 Variable 列表（未解包）")
args6 = test_pack(var_list)
# 尝试取 data（会报错！因为 args6[0] 是列表，不是 Variable）
try:
    print(args6[0].data)
except AttributeError as e:
    print(f"报错原因: {e} → args6[0] 是列表，不是 Variable")

# 场景7：解包传 Variable 列表（加*）
print("\n【场景7】解包传 Variable 列表（加*）")
args7 = test_pack(*var_list)
# 正常取 data（args7[0] 是 Variable 实例）
print(f"args7[0].data = {args7[0].data}")
print(f"args7[1].data = {args7[1].data}")

# ===================== 实际开发：*args 的常用场景 =====================
print("\n===== 实际应用：*args 简化函数调用 =====")
# 场景8：求和函数（支持任意个数参数）
def my_sum(*args):
    total = 0
    for num in args:
        total += num
    return total

print("\n【场景8】求和函数（传任意个数参数）")
print(f"my_sum(1,2,3) = {my_sum(1,2,3)}")
print(f"my_sum(10,20) = {my_sum(10,20)}")
print(f"my_sum() = {my_sum()}")  # 空参数返回0

# 场景9：解包列表/元组调用函数
print("\n【场景9】解包列表调用求和函数")
nums = [100, 200, 300]
print(f"my_sum(*nums) = {my_sum(*nums)}")  # 等价于 my_sum(100,200,300)

# ===================== 总结：核心规则（新手背记） =====================
print("\n===== 核心规则总结 =====")
rules = [
    "1. *args 作用：把「多个独立参数」打包成「元组」（永远是元组）",
    "2. *列表 作用：把「列表/元组」解包成「多个独立参数」",
    "3. 直接传列表 → args = (列表,)（元组里只有1个元素：列表）",
    "4. 解包传列表 → args = (列表元素1, 列表元素2, ...)（元组长度=列表长度）",
    "5. 易错点：直接传列表时，args[0] 是列表本身，不是列表里的元素"
]
for rule in rules:
    print(rule)