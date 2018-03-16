import numpy as np
print(np.__version__)    #输出版本号

#creating_ndarrays
print('使用普通一维数组生成NumPy一维数组')
data = [6, 7.5, 8, 0, 1]
arr = np.array(data)
print(arr)
print('打印元素类型')
print(arr.dtype)

print('使用普通二维数组生成NumPy二维数组')
data = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr = np.array(data)
print(arr)
print('打印数组维度(几行几列）')
print(arr.shape)

print('使用zeros产生全0数组') #还有一个ones函数用来产生全1数组，zeros_like和one_like以另一个数组为参数，根据它创建新数组
print(np.zeros(10))    # 生成包含10个0的一维数组
print(np.zeros((3, 6)))  # 生成3*6的二维数组

print('使用empty产生一个数组的内存空间，但不填充任何值')
print(np.empty((2, 3, 2)))  # 生成2*3*2的三维数组，所有元素未初始化。

print('使用arange生成连续元素')
print(np.arange(15))     # [0, 1, 2, ..., 14])

print('使用eye产生N*N单位矩阵')    #eye还有其他参数，cdentity函数有类似效果
print(np.eye(2,dtype=int))

#data_types_for_ndarray
print('生成数组时指定数据类型')
arr = np.array([1, 2, 3], dtype = np.float64)
print(arr.dtype)
arr = np.array([1, 2, 3], dtype = np.int32)
print(arr.dtype)

print('使用astype复制数组并转换数据类型')
int_arr = np.array([1, 2, 3, 4, 5])
float_arr = int_arr.astype(np.float)
print(int_arr.dtype)
print(float_arr.dtype)

print('使用astype将float转换为int时小数部分被舍弃')
float_arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
int_arr = float_arr.astype(dtype = np.int)
print(int_arr)

print('使用astype把字符串转换为数组，如果失败抛出异常。')
str_arr = np.array(['1.25', '-9.6', '42'], dtype = np.string_)
float_arr = str_arr.astype(dtype = np.float)
print(float_arr)

print('astype使用其它数组的数据类型作为参数')
int_arr = np.arange(10)
float_arr = np.array([.23, 0.270, .357, 0.44, 0.5], dtype = np.float64)
float_arr2 = int_arr.astype(float_arr.dtype)
print(float_arr2.dtype)
print(int_arr.dtype)  # astype做了复制，数组本身不变。

#operations_between_arrays_and_scalars
# 数组乘法／减法，对应元素相乘／相减。
arr = np.array([[1.0, 2.0, 3.0], [4., 5., 6.]])
print(arr * arr)
print(arr - arr)

# 标量操作作用在数组的每个元素上
arr = np.array([[1.0, 2.0, 3.0], [4., 5., 6.]])
print(1 / arr)
print(arr ** 0.5) # 开根号

# basic_indexing_and_slicing  索引和切片
# 通过索引访问二维数组某一行或某个元素
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr[2])
print(arr[0][2])
print(arr[0, 2])  # 普通Python数组不能用。这里的arr[0][2]和arr[0,2]是一样的

# 对更高维数组的访问和操作
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])   # 这是3维数组
print(arr[0])  # 结果是个2维数组
print(arr[1, 0])
old_values = arr[0].copy()  # 复制arr[0]的值
arr[0] = 42  # 把arr[0]所有的元素都设置为同一个值
print(arr)
arr[0] = old_values  # 把原来的数组写回去
print(arr)

print('使用切片访问和操作数组')
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(arr[1:6])  # 打印元素arr[1]到arr[5]
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr[:2])  # 打印第0、1行
print(arr[:2, 1:])  # 打印第0、1行的第1、2列
print(arr[:, :1])  # 打印第一列的所有元素
arr[:2, 1:] = 0  # 第0、1行，第1、2列的元素设置为0
print(arr)

# boolean_indexing
import numpy.random as np_random
print('使用布尔数组作为索引')
name_arr = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
rnd_arr = np_random.randn(7, 4) # 随机7*4数组
print(rnd_arr)
print(name_arr == 'Bob')  # 返回布尔数组，元素等于'Bob'为True，否则False。
print(rnd_arr[name_arr == 'Bob'])  # 利用布尔数组选择行,把布尔数组中true的那几行打印出来
print(rnd_arr[name_arr == 'Bob', :2])  # 增加限制打印列的范围
print(rnd_arr[~(name_arr == 'Bob')])  # 对布尔数组的内容取反
mask_arr = (name_arr == 'Bob') | (name_arr == 'Will')  # 逻辑运算混合结果
print(rnd_arr[mask_arr])
rnd_arr[name_arr != 'Joe'] = 7  # 先布尔数组选择行，然后把每行的元素设置为7。
print(rnd_arr)

#fancy_indexing
print('Fancy Indexing: 使用整数数组作为索引')
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i     # 每行的每个数都等于数所在的行数
print(arr)
print(arr[[4, 3, 0, 6]])  # 打印arr[4]、arr[3]、arr[0]和arr[6]。
print(arr[[-3, -5, -7]])  # 打印arr[3]、arr[5]和arr[-7]行
arr = np.arange(32).reshape((8, 4))  # 通过reshape变换成二维数组
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]])  # 打印arr[1, 0]、arr[5, 3]，arr[7, 1]和arr[2, 2]
print(arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]])  # 1572行的0312列
# 上一行，也就是先把1，5，7，2行提取出来组成一个4*4的二维数组，再把这个二维数组的列按照0312的顺序重新排列
print(arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])  # 可读性更好的写法

# transposing_arrays_and_swapping_axes
import numpy.random as np_random
print('转置矩阵')
arr = np.arange(15).reshape((3, 5))
print(arr)
print(arr.T)
print('转置矩阵做点积')
arr = np_random.randn(6, 3)
print(np.dot(arr.T, arr))
print('高维矩阵转换')
arr = np.arange(16).reshape((2, 2, 4))   #有两层，每层2*4数组
print(arr)
'''
详细解释：
arr数组的内容为
- a[0][0] = [0, 1, 2, 3]
- a[0][1] = [4, 5, 6, 7]
- a[1][0] = [8, 9, 10, 11]
- a[1][1] = [12, 13, 14, 15]
transpose的参数为坐标，正常顺序为(0, 1, 2, ... , n - 1)，
现在传入的为(1, 0, 2)代表a[x][y][z] = a[y][x][z]，第0个和第1个坐标互换。
- a'[0][0] = a[0][0] = [0, 1, 2, 3]
- a'[0][1] = a[1][0] = [8, 9, 10, 11]
- a'[1][0] = a[0][1] = [4, 5, 6, 7]
- a'[1][1] = a[1][1] = [12, 13, 14, 15]
'''
print(arr.transpose((1, 0, 2)))  # 注意transpose和swapaxes的区别！！！
print(arr.swapaxes(1, 2))  # 直接交换第1和第2个坐标，轴对换

# universal_functions
import numpy.random as np_random
print('求平方根')
arr = np.arange(10)
print(np.sqrt(arr))  # 等价于arr**0.5
print('数组比较')
x = np_random.randn(8)
y = np_random.randn(8)
print(x)
print(y)
print(np.maximum(x, y))
print('使用modf函数把浮点数分解成整数和小数部分')
arr = np_random.randn(7) * 5  # 统一乘5
print(np.modf(arr))   # 将数组的小数部分与整数部分以两个独立数组的形式返还

# intro  用数组表达式代替循环的做法，通常被称为矢量化，这样会快一些
import matplotlib.pyplot as plt
import pylab
points = np.arange(-5, 5, 0.01)  # 生成100个点
xs, ys = np.meshgrid(points, points)  # xs, ys互为转置矩阵,meshgrid的用法需理解
print(xs)
print(ys)
z = np.sqrt(xs ** 2 + ys ** 2)    # 直接对整个数组的各个数进行平方和相加，不需要循环
print(z)
plt.imshow(z, cmap=plt.cm.gray)    # 绘画灰度图，是关于z的
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
pylab.show()

# expressing_conditional_logic_as_array_operations
import numpy.random as np_random
'''
关于zip函数的一点解释，zip可以接受任意多参数，然后重新组合成1个tuple列表。
zip([1, 2, 3], [4, 5, 6], [7, 8, 9])
返回结果：[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
'''
print('通过真值表选择元素')
x_arr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
y_arr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = [(x if c else y) for x, y, c in zip(x_arr, y_arr, cond)]  # 通过列表推到实现
# 上一行，是一个三元表达式，意思是如果c（也就是cond))为真true就选x给result，如果c为假false就选y给result
# zip可以理解为打包，把x,y,c打包在一起，这样就可以直接用一个for in语句，否则的话要用三个for in语句
print(result)
print(np.where(cond, x_arr, y_arr))  # 使用NumPy的where函数,这里的功能类似于一个三元表达式
print('更多where的例子')
arr = np_random.randn(4, 4)
print(arr)
print(np.where(arr > 0, 2, -2))   # 使用NumPy的where函数,这里的功能类似于一个三元表达式
print(np.where(arr > 0, 2, arr))
print('where嵌套')
cond_1 = np.array([True, False, True, True, False])
cond_2 = np.array([False, True, False, True, False])
# 传统代码如下
result = []
for i in range(len(cond)):
    if cond_1[i] and cond_2[i]:
        result.append(0)
    elif cond_1[i]:
        result.append(1)
    elif cond_2[i]:
        result.append(2)
    else:
        result.append(3)
print(result)
# np版本代码
result = np.where(cond_1 & cond_2, 0, \
          np.where(cond_1, 1, np.where(cond_2, 2, 3)))    # \ 表示下一行接着上一行
print(result)

# mathematical_and_statistical_methods
import numpy.random as np_random
print('求和，求平均')
arr = np.random.randn(5, 4)
print(arr)
print(arr.mean())   # 求算术平均数
print(arr.sum())    # 求和
arr=np.array([[1,2],[3,4]])
print(arr.mean(axis = 1))  # 对每一列的元素相加后求平均，1表示列
print(arr.sum(0))  # 对每一行元素求和，axis可以省略，0表示行
'''
cumsum:
- 按列操作：a[i][j] += a[i - 1][j]，把所有i都循环一遍，比如i=3时，那么后来的第3行为原先的第1，2，3行相加
- 按行操作：a[i][j] *= a[i][j - 1]
cumprod:
- 按列操作：a[i][j] += a[i - 1][j]
- 按行操作：a[i][j] *= a[i][j - 1]
'''
print('cunsum和cumprod函数演示')
arr =(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
print(arr.cumsum(0))    # 0表示行
print(arr.cumprod(1))   # 1表示列

# methods_for_boolean_arrays  bool型数组的处理
print('对正数求和')
arr = np_random.randn(100)
print((arr > 0).sum())
print('对数组逻辑操作')
bools = np.array([False, False, True, False])
print(bools.any())  # 有一个为True则返回True
print(bools.all())  # 有一个为False则返回False

#  sorting  利用数组进行数据处理 排序
print('一维数组排序')
arr = np_random.randn(8)
arr.sort()
print(arr)
print('二维数组排序')
arr = np_random.randn(5, 3)
print(arr)
arr.sort(0)  # 对每一列元素做排序,1表示对行进行排序，0表示对列进行排序
print(arr)
print('找位置在5%的数字')
large_arr = np_random.randn(1000)
large_arr.sort()
print(large_arr[int(0.05 * len(large_arr))])

# unique_and_other_set_logic  利用数组进行数据处理 去重以及其它集合运算
print('用unique函数去重')
names = np.array([ 'Joe','Bob', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(sorted(set(names)))  # 传统Python做法   没看懂？？？
print(np.unique(names))
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
print(np.unique(ints))
print('查找数组元素是否在另一数组')
values = np.array([6, 0, 0, 3, 2, 5, 6])
print(np.in1d(values, [2, 3, 6]))

# linear_algebra  线性代数，矩阵
from numpy.linalg import inv, qr
print('矩阵乘法')
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print(x.dot(y))    # 矩阵乘法
print(np.dot(x, np.ones(3)))   # 矩阵乘法
x = np_random.randn(5, 5)
print('矩阵求逆')
mat = x.T.dot(x)
print(inv(mat))  # 矩阵求逆
print(mat.dot(inv(mat)))  # 与逆矩阵相乘，得到单位矩阵。
print('矩阵消元')
print(mat)
q, r = qr(mat)   # 计算QR分解，q和r都有自己的含义
print(q)
print(r)

# random_number_generation 随机数生成
from random import normalvariate
print('正态分布随机数')
samples = np.random.normal(size=(4, 4))
print(samples)
print('批量按正态分布生成0到1的随机数')
N = 10
print([normalvariate(0, 1) for _ in range(N)])
print(np.random.normal(size = N))  # 与上面代码等价

# reshaping_arrays  高级应用 数组重塑
print("将一维数组转换为二维数组")
arr = np.arange(8)
print(arr.reshape((4, 2)))
print(arr.reshape((4, 2)).reshape((2, 4))) # 支持链式操作
print("维度大小自动推导")
arr = np.arange(15)
print(arr.reshape((5, -1)))    # -1表示自动推导维度大小
print("获取维度信息并应用")
other_arr = np.ones((3, 5))
print(other_arr.shape)
print(arr.reshape(other_arr.shape))
print("高维数组拉平")
arr = np.arange(15).reshape((5, 3))
print(arr.ravel())    # ravel的意思就是解开，弄清

# concatenating_and_splitting_arrays 高级应用 数组的合并和拆分
# 合并(连接，堆叠）的方式，1、concatenate 2、vstack hstack 3、r_ c_
print('连接两个二维数组')
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
print(np.concatenate([arr1, arr2], axis = 0))
# 按行连接,自己的理解，可以把arr1看成一个整体，看成一行，然后再把arr2看成一个整体放到第二行
print(np.concatenate([arr1, arr2], axis = 1) )
# 按列连接，自己的理解，可以把arr1看成一个整体，看成一列，然后再把arr2看成一个整体放到第二列
# 所谓堆叠，参考叠盘子。。。连接的另一种表述
print('垂直stack与水平stack')
print(np.vstack((arr1, arr2))) # 垂直堆叠
print(np.hstack((arr1, arr2))) # 水平堆叠
print('拆分数组')
arr = np_random.randn(5, 5)
print(arr)
print('水平拆分')
first, second, third = np.split(arr, [1, 3], axis = 0)  #把1和3作为分割行，也就是0，1和2，3和4行这样分割，0表示按行分割
# 如果不是写[1,3]，而是写一个整数，那就会把所有行平均分成整数份。
print('first')
print(first)
print('second')
print(second)
print('third')
print(third)
print('垂直拆分')
first, second, third = np.split(arr, [1, 3], axis = 1)  #1表示按列分割
print('first')
print(first)
print('second')
print(second)
print('third')
print(third)
# 堆叠辅助类 r_和c_函数也可以用合并（连接）
arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = np_random.randn(3, 2)
print('r_用于按行堆叠')
print(np.r_[arr1, arr2])
print('c_用于按列堆叠')
print(np.c_[np.r_[arr1, arr2], arr])
print('切片直接转为数组')
print(np.c_[1:6, -10:-5])  # 直接就拼成一个二元数组，1:6为一列，-10:-5为一列

# repeating_elements  高级应用 元素的重复操作
print('Repeat: 按元素')
arr = np.arange(3)
print(arr.repeat(3))  # 每个元素重复3次
print(arr.repeat([2, 3, 4])) # 3个元素，分别复制2, 3, 4次。长度要匹配！
print('Repeat，指定轴')
arr = np_random.randn(2, 2)
print(arr)
print(arr.repeat(2, axis = 0)) # 按行repeat
print(arr.repeat(2, axis = 1)) # 按列repeat
print('Tile: 参考贴瓷砖')    # 把原先的arr看成一个整体（一个瓷砖）
print(np.tile(arr, 2))
print(np.tile(arr, (2, 3)))  # 指定每个轴的tile次数

# fancy_indexing_equivalents  高级应用 花式索引的等价函数    ？？？？这里还没搞懂
print('Fancy Indexing例子代码')
arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
print(arr[inds])
print('使用take')
print(arr.take(inds))  # take函数的作用跟索引一样
print('使用put更新内容')
arr.put(inds, 50)
print(arr)
arr.put(inds, [70, 10, 20, 60])
print(arr)
print('take，指定轴')
arr = np_random.randn(2, 4)
inds = [2, 0, 2, 1]
print(arr)
print(arr.take(inds, axis = 1))  # 按列take

# example_random_walks
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as np_random
import pylab
print('模拟随机游走')
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)    # randint 生成在半开半闭区间[0,2)上离散均匀分布的整数值,也就是1或者0
print(draws)
steps = np.where(draws > 0, 1, -1)  # 如果drawsd大于0，则为1，如果draws不大于0，则为-1
print(steps)
walk = steps.cumsum()
# cumsum表示所有元素的累积和，walk也是一个数组，walk数组的一个元素就是steps数组的对应元素和该元素之前的元素累加的和的值，所以walk值有大有小
print(walk)
# 画图
plt.title('Random Walk')
limit = max(abs(min(walk)), abs(max(walk)))  # 计算纵轴的上下限
plt.axis([0, 1000, -limit, limit])  # 取坐标轴纵轴的上下限，横轴的范围
x = np.linspace(0, 1000, 1000)  # 把0到1000平均分成1000个数字(999份等间隔空间）（等差数列），因为是从0开始的，所以不是刚好凑整的
# 如果是np.linspace(1,1000,1000)，这样刚好是整数，1，2，3，4...1000
print(x)
plt.plot(x, walk, 'g-')  # x为横轴，walk为纵轴，
plt.show()
'''
.plot函数
plt.plot(x, y, format_string, **kwargs): x为x轴数据，可为列表或数组；y同理；format_string为控制曲线的格式字符串， **kwargs为第二组或更多的（x, y, format_string）
format_string: 由 颜色字符、风格字符和标记字符组成。
颜色字符：‘b’蓝色  ；‘#008000’RGB某颜色；‘0.8’灰度值字符串
风格字符：‘-’实线；‘--’破折线； ‘-.’点划线； ‘：’虚线 ； ‘’‘’无线条
标记字符：‘.’点标记  ‘o’ 实心圈 ‘v’倒三角  ‘^’上三角
eg： plt.plot(a, a*1.5, ‘go-’,    a, a*2, ‘*’)   第二条无曲线，只有点
'''

#下面这样是一样的
arr = np.arange(10)
out1 = np.where(arr % 2 == 1, -1, arr)  # 数组级别的
out2 = [(-1 if arr[i] % 2 == 1 else arr[i]) for i in range(10)]    # 元素级别的
# 上一行，只要if前面和else后面的东西级别一样就可以，可以两个都是数组，也可以两个都是元素，但不要一个是数组，一个是元素。
print(out1)
print(out2)






