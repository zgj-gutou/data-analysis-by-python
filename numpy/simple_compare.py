#用于比较不同的数据结构list,array,numpy.array的运算速度问题
import timeit      #python中的计时器

#这里三个双引号里面包着的是字符串表达式，作为后面的形参
#用for循环计算和
common_for ="""
for d in data:  
    s += d
"""
#用python自带的sum函数计算和
common_sum ="""
sum(data)
"""
#用numpy的sum函数计算和
common_numpy_sum ="""
numpy.sum(data)
"""

#list_setup是用list
def timeit_list(n,loops):
    list_setup = """
import numpy
data = [1] * {}
s = 0
""".format(n)
    print('list:')
    # number表示common_for执行的次数，记的时间也是common_for执行的时间,list_setup用于初始化,这里指给data和s初始化，下同
    print(timeit.timeit(common_for,list_setup,number = loops))
    print(timeit.timeit(common_sum,list_setup,number = loops))
    print(timeit.timeit(common_numpy_sum,list_setup,number = loops))

#array_setup是用python自己的array
def timeit_array(n, loops):
    array_setup = """
import numpy
import array
data = array.array('L', [1] * {})
s = 0
""".format(n)
    print('array:')
    print(timeit.timeit(common_for, array_setup, number = loops))
    print(timeit.timeit(common_sum, array_setup, number = loops))
    print(timeit.timeit(common_numpy_sum, array_setup, number = loops))

#numpy_setup使用numpy的array
def timeit_numpy(n, loops):
    numpy_setup = """
import numpy
data = numpy.array([1] * {})
s = 0
""".format(n)
    print('numpy:')
    print(timeit.timeit(common_for, numpy_setup, number = loops))
    print(timeit.timeit(common_sum, numpy_setup, number = loops))
    print(timeit.timeit(common_numpy_sum, numpy_setup, number = loops))

if __name__ == '__main__':
    timeit_list(50000,500)     #数组大小是50000，每次500次循环
    timeit_array(50000,500)
    timeit_numpy(50000,500)
