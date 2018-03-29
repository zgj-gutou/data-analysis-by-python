import pandas as pd
import numpy as np
from pandas import Series, DataFrame

# series
print('用数组生成Series')
obj = Series([4, 7, -5, 3])   # 不指定索引，就有默认索引
print(obj)
print(obj.values)
print(obj.index)
print('指定Series的index')
obj2 = Series([4, 7, -5, 3], index = ['d', 'b', 'a', 'c'])   # 记住格式
print(obj2)
print(obj2.index)
print(obj2['a'])
obj2['d'] = 6
print(obj2[['c', 'a', 'd']])   # 花式索引，注意这里有两个[]包住索引，如果只查看某一个索引对应的数据话，只要一个[]就可以。
print(obj2[obj2 > 0])  # 找出大于0的元素
print('b' in obj2) # 判断索引是否存在
print('e' in obj2)
print('使用字典生成Series')
sdata = {'Ohio':45000, 'Texas':71000, 'Oregon':16000, 'Utah':5000}
obj3 = Series(sdata)
print(obj3)
print('使用字典生成Series，并额外指定index，不匹配部分为NaN。')
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index = states)     #  California匹配不到值的时候，会显示NaN
print(obj4)
print('Series相加，相同索引部分相加。')
print(obj3 + obj4)
print('指定Series及其索引的名字')
obj4.name = 'population'
obj4.index.name = 'state'
print(obj4)
print('替换index')
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)

# dataframe
print('用字典生成DataFrame，key为列的名字。')    # 没有指定行索引
data = {'state':['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year':[2000, 2001, 2002, 2001, 2002],
        'pop':[1.5, 1.7, 3.6, 2.4, 2.9]}
print(DataFrame(data))
print(DataFrame(data, columns = ['year', 'state', 'pop'])) # 指定列顺序
print('指定索引，在列中指定不存在的列，默认数据用NaN。')    # 下面debt的列不存在
frame2 = DataFrame(data,                                  # DataFrame函数的参数可以带columns和index，也可以不带
                    columns = ['year', 'state', 'pop', 'debt'],   # columns的东西在每一列
                    index = ['one', 'two', 'three', 'four', 'five'])  # index的东西在每一行
print(frame2)
print(frame2['state'])    # 访问列索引方式一
print(frame2.year)     # 访问列索引方式二
print(frame2.ix['three'])   # 访问行索引，一定要加ix，用来区分行索引和列索引
frame2['debt'] = 16.5 # 修改一整列
print(frame2)
frame2.debt = np.arange(5)  # 用numpy数组修改元素
print(frame2)
print('用Series指定要修改的索引及其对应的值，没有指定的默认数据用NaN。')
val = Series([-1.2, -1.5, -1.7], index = ['two', 'four', 'five'])
frame2['debt'] = val
print(frame2)
print('赋值给新列')  # 下面一行是bool型的赋值
frame2['eastern'] = (frame2.state == 'Ohio')  # 如果state等于Ohio为True
print(frame2)
print(frame2.columns)   # 打印列的名字
print(frame2.index)  # 打印行的名字
print('DataFrame转置')
pop = {'Nevada':{2001:2.4, 2002:2.9},
        'Ohio':{2000:1.5, 2001:1.7, 2002:3.6}}   # 这里的2000，2001，2002即index
frame3 = DataFrame(pop)
print(frame3)
print(frame3.T)   # 转置，行列反过来
print('指定索引顺序，以及使用切片初始化数据。')
print(DataFrame(pop, index = [2001, 2002, 2003]))   # 指定索引顺序
pdata = {'Ohio':frame3['Ohio'][:-1], 'Nevada':frame3['Nevada'][:2]}  # 用切片来初始化数据，注意原先的frame3并没有改变
print(DataFrame(pdata))
print(frame3)
print('指定索引和列的名称')
frame3.index.name = 'year'  # 行的名字
frame3.columns.name = 'state'  # 列的名字
print(frame3)
print(frame3.values)   # 不带行列的信息
print(frame2.values)

# index_objects
import sys
from pandas import Index
print('获取index')
obj = Series(range(3), index = ['a', 'b', 'c'])
index = obj.index
print(index[1:])
try:
    index[1] = 'd'  # index对象read only ，index对象不可修改，但index是可以修改的，用reindex
except:
    print(sys.exc_info()[0])
print('使用Index对象')
index = Index(np.arange(3))  # 生成index
obj2 = Series([1.5, -2.5, 0], index = index)
print(obj2)
print(obj2.index is index)
print('判断列和索引是否存在指定的东西')
pop = {'Nevada':{20001:2.4, 2002:2.9},
        'Ohio':{2000:1.5, 2001:1.7, 2002:3.6}}
frame3 = DataFrame(pop)
print(frame3)
print('Ohio' in frame3.columns)
print('2003' in frame3.index)

# reindexing  重新索引
print('重新指定索引及顺序')
obj = Series([4.5, 7.2, -5.3, 3.6], index = ['d', 'b', 'a', 'c'])
print(obj)
obj2 = obj.reindex(['a', 'b', 'd', 'c', 'e'])
print(obj2)
print(obj.reindex(['a', 'b', 'd', 'c', 'e'], fill_value = 0))  # 指定不存在元素的默认值
print('重新指定索引并指定填元素充方法')
obj3 = Series(['blue', 'purple', 'yellow'], index = [0, 2, 4])
print(obj3)
print(obj3.reindex(range(6), method = 'bfill'))
#   ffill：插值时向前取值,用前面一行的值填充  bfill：插值时向后取值，用后面一行的值填充
print('对DataFrame重新指定索引')
frame = DataFrame(np.arange(9).reshape(3, 3),
                  index = ['a', 'c', 'd'],
                  columns = ['Ohio', 'Texas', 'California'])
print(frame)
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
print(frame2)
print('重新指定column')
states = ['Texas', 'Utah', 'California']
print(frame.reindex(columns = states))
print('对DataFrame重新指定索引并指定填元素充方法')
''' # 下面被注释掉的这个运行有问题，可能是pandas或者python的版本问题造成的
print(frame.reindex(index = ['a', 'b', 'c', 'd'],
                    method = 'ffill',
                    columns = states))
'''
print(frame.ix[['a', 'b', 'd', 'c'], states])   # 重新指定索引，没有填充元素

# dropping_entries_from_an_axis  丢弃指定轴上的项
print('Series根据索引删除元素')
obj = Series(np.arange(5.), index = ['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
print(new_obj)
print(obj.drop(['d', 'c']))    # 注意几个索引要像数组一样用[]包起来
print('DataFrame删除元素，可指定索引或列。')
data = DataFrame(np.arange(16).reshape((4, 4)),
                  index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                  columns = ['one', 'two', 'three', 'four'])
print(data)
print(data.drop(['Colorado', 'Ohio']))
print(data.drop('two', axis = 1))    # axis=1 表示列
print(data.drop(['two', 'four'], axis = 1))

# indexing_selection_and_filtering 索引、选取和过滤
print('Series的索引，默认数字索引可以工作。')
obj = Series(np.arange(4.), index = ['a', 'b', 'c', 'd'])
print(obj['b'])
print(obj[3])
print(obj[[1, 3]])  # 1和3是索引，1和3单独写出来用逗号隔开说明是指定的1和3索引，如果用冒号隔开说明是一个区间，这样的话就不需要加[]
# 注意几个索引要像数组一样用[]包起来，然后本来就需要一个[]的，所以有两个[]
print(obj[obj < 2])
print('Series的数组切片')
print(obj['b':'c'])  # 闭区间，'b':'c'是一个区间，所以不需要再加[]，只需要一个[]就够了
obj['b':'c'] = 5
print(obj)
print('DataFrame的索引')
data = DataFrame(np.arange(16).reshape((4, 4)),
                  index = ['Ohio', 'Colorado', 'Utah', 'New York'],
                  columns = ['one', 'two', 'three', 'four'])
print(data)
print(data['two'])  # 打印列
print(data[['three', 'one']])  # 列标签为three和one的那两列
print(data[:2])   # 表示第0行和第1行
print(data.ix['Colorado', ['two', 'three']])  # 指定索引和列
print(data.ix[['Colorado', 'Utah'], [3, 0, 1]])  # 非数字的索引或列，可以用默认的数字匹配对应
print(data.ix[2])  # 打印第2行（有第0行时的第2行）  ix表示通过行号索引或通过行标签索引
print(data.ix[:'Utah', 'two'])  # 从开始到Utah，第2列。
print('根据条件选择')   # 用条件表示索引
print(data[data.three > 5])
print(data < 5)  # 打印True或者False
data[data < 5] = 0
print(data)

# arithmetic_and_data_alignment 算术运算和数据对齐
print('加法')
s1 = Series([7.3, -2.5, 3.4, 1.5], index = ['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index = ['a', 'c', 'e', 'f', 'g'])
print(s1)
print(s2)
print(s1 + s2)
print('DataFrame加法，索引和列都必须匹配。')   # 如果不匹配，索引和列会变成并集，但值变成了NaN
df1 = DataFrame(np.arange(9.).reshape((3, 3)),
                columns = list('bcd'),
                index = ['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12).reshape((4, 3)),
                columns = list('bde'),
                index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
print(df1)
print(df2)
print(df1 + df2)
print('数据填充')
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns = list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns = list('abcde'))
print(df1)
print(df2)
print(df1.add(df2, fill_value = 0))
print(df1.reindex(columns = df2.columns, fill_value = 0))   # 把NaN的地方的值变成0
print('DataFrame与Series之间的操作')
arr = np.arange(12.).reshape((3, 4))
print(arr)
print(arr[0])
print(arr - arr[0])
frame = DataFrame(np.arange(12).reshape((4, 3)),
                  columns = list('bde'),
                  index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
print(frame)
print(series)
print(frame - series)
series2 = Series(range(3), index = list('bef'))
print(frame + series2)  # 任意一个值和缺失值相加的话，结果还是缺失值
series3 = frame['d']
print(frame.sub(series3, axis = 0))  # 按列减

# function_application_and_mapping 函数应用和映射
print('函数')
frame = DataFrame(np.random.randn(4, 3),
                  columns = list('bde'),
                  index = ['Utah', 'Ohio', 'Texas', 'Oregon'])
print(frame)
print(np.abs(frame))
print('lambda以及应用')
f = lambda x: x.max() - x.min()
print(frame.apply(f))  # 应用这个f匿名函数，这里是每一列的最大值减最小值
print(frame.apply(f, axis = 1))  # 对行进行操作
def f(x):
    return Series([x.min(), x.max()], index = ['min', 'max'])
print(frame.apply(f))
print('applymap和map')
_format = lambda x: '%.2f' % x  # 以两位小数的格式打印  后面那个%用来分离格式和输出的数
print(frame.applymap(_format))   # applymap作用到frame的每一个元素上去
print(frame['e'].map(_format))  # frame['e']实际上是一个series类型，map作用到series的每一个元素上

# sorting_and_ranking 排序和排名
print('根据索引排序，对于DataFrame可以指定轴。')
obj = Series(range(4), index = ['d', 'a', 'b', 'c'])
print(obj.sort_index())
frame = DataFrame(np.arange(8).reshape((2, 4)),
                  index = ['three', 'one'],
                  columns = list('dabc'))
print(frame.sort_index())  # 默认axis=0，表示行
print(frame.sort_index(axis = 1)) # 表示列
print(frame.sort_index(axis = 1, ascending = False)) # 降序
print('根据值排序')
obj = Series([4, 7, -3, 2])
print(obj.sort_values()) # order已淘汰
print('DataFrame指定列排序')
frame = DataFrame({'b':[4, 7, -3, 2], 'a':[0, 1, 0, 1]})
print(frame)  # 下面的sort_values可以指定某行某列
print(frame.sort_values(by = 'b')) # sort_index(by = ...)已淘汰
print(frame.sort_values(by = ['a', 'b']))  # 排序优先级，先按列a排序，如果一样的列a中两个值一样的话，再按列b排序
print('rank，求排名的平均位置(从1开始)')
obj = Series([7, -5, 7, 4, 2, 0, 4])
# 对应排名：-5(1), 0(2), 2(3), 4(4), 4(5), 7(6), 7(7)
print(obj.rank())
print(obj.rank(method = 'first'))  # 去第一次出现，不求平均值。
print(obj.rank(ascending = False, method = 'max')) # 逆序（降序），并取最大值。所以-5的rank是7.
frame = DataFrame({'b':[4.3, 7, -3, 2],
                  'a':[0, 1, 0, 1],
                  'c':[-2, 5, 8, -2.5]})
print(frame)
print(frame.rank(axis = 1))  # 表示对列

# axis_indexes_with_duplicate_values 带有重复值的索引
print('重复的索引')
obj = Series(range(5), index = ['a', 'a', 'b', 'b', 'c'])
print(obj.index.is_unique) # 判断是否有重复索引
# print(obj['a'][0])
# print(obj.a[1])
df = DataFrame(np.random.randn(4, 3), index = ['a', 'a', 'b', 'b'])
print(df)
print(df.ix['b'].ix[0])  # 表示b索引的那些行中的第一行
print(df.ix['b'].ix[1])

# intro 汇总和计算描述统计
print('求和')
df = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
              index = ['a', 'b', 'c', 'd'],
              columns = ['one', 'two'])
print(df)
print(df.sum())  # 按列求和
print(df.sum(axis = 1))  # 按行求和
print('平均数')
print(df.mean(axis = 1, skipna = False))
print(df.mean(axis = 1))
print('其它')
print(df.idxmax())
print(df.cumsum())
print(df.describe())  # 对数字型的东西的统计
obj = Series(['a', 'a', 'b', 'c'] * 4)
print(obj.describe())   # 对非数字型的东西的统计

'''
# correlation_and_covariance 汇总和计算描述统计 相关系数与协方差  量化交易时会用到这个
import pandas.io.data as web
print('相关性与协方差')  # 协方差：https://zh.wikipedia.org/wiki/%E5%8D%8F%E6%96%B9%E5%B7%AE
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
    all_data[ticker] = web.get_data_yahoo(ticker, '4/1/2016', '7/15/2015')
    price = DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
    volume = DataFrame({tic: data['Volume'] for tic, data in all_data.iteritems()})
returns = price.pct_change()
print(returns.tail())
print(returns.MSFT.corr(returns.IBM))
print(returns.corr())  # 相关性，自己和自己的相关性总是1
print(returns.cov()) # 协方差
print(returns.corrwith(returns.IBM))
print(returns.corrwith(returns.volume))
'''

# unique_values_value_counts_and_membership 汇总和计算描述统计 唯一值以及成员资格
print('去重')
obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
print(obj.unique())
print(obj.value_counts())
print('判断元素存在')
mask = obj.isin(['b', 'c'])
print(mask)
print(obj[mask]) #只打印元素b和c
data = DataFrame({'Qu1':[1, 3, 4, 3, 4],
                  'Qu2':[2, 3, 1, 2, 3],
                  'Qu3':[1, 5, 2, 4, 4]})
print(data)
print(data.apply(pd.value_counts).fillna(0))
print(data.apply(pd.value_counts, axis = 1).fillna(0))

# intro 处理缺失数据
print('作为null处理的值')
string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
print(string_data)
print(string_data.isnull())
string_data[0] = None
print(string_data.isnull())

# filtering_out_missing_data 处理缺失数据 滤除缺失数据
from numpy import nan as NA
print('丢弃NA')
data = Series([1, NA, 3.5, NA, 7])
print(data.dropna())
print(data[data.notnull()])
print('DataFrame对丢弃NA的处理')
data = DataFrame([[1., 6.5, 3.], [1., NA, NA],
                  [NA, NA, NA], [NA, 6.5, 3.]])
print(data.dropna())  # 默认只要某行有NA就全部删除
print(data.dropna(how = 'all'))  # 全部为NA才删除
data[4] = NA  # 新增一列
print(data)
print(data.dropna(axis = 1, how = 'all'))  # 1表示列
data = DataFrame(np.random.randn(7, 3))
data.ix[:4, 1] = NA
data.ix[:2, 2] = NA
print(data)
print(data.dropna(thresh = 2))  # 每行至少要有2个非NA元素

# filling_in_missing_data 处理缺失数据 填充缺失数据
print('填充0')
df = DataFrame(np.random.randn(7, 3))
df.ix[:4, 1] = NA
df.ix[:2, 2] = NA
print(df.fillna(0))
df.fillna(0, inplace = True)
print(df)
print('不同行列填充不同的值')
print(df.fillna({1:0.5, 3:-1}))  # 第3列不存在
print('不同的填充方式')
df = DataFrame(np.random.randn(6, 3))
df.ix[2:, 1] = NA
df.ix[4:, 2] = NA
print(df)
print(df.fillna(method = 'ffill'))
print(df.fillna(method = 'ffill', limit = 2))
print('用统计数据填充')
data = Series([1., NA, 3.5, NA, 7])
print(data.fillna(data.mean()))

# intro 层次化索引
from pandas import Series, DataFrame, MultiIndex
print('Series的层次索引')
data = Series(np.random.randn(10),
              index = [['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'],
                       [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
print(data)
print(data.index)
print(data.b)
print(data['b':'c'])
print(data[:2])   # 打印前两行
print(data.unstack())  # unstack之后就是一个dataframe
print(data.unstack().stack())  # stack之后是一个series
print('DataFrame的层次索引')
frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns = [['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
print(frame)
frame.index.names = ['key1', 'key2']
frame.columns.names = ['state', 'color']
print(frame)
print(frame.ix['a', 1])   # a表示key1,1表示key2，这条语句表示key1为a，key2为1的那一行
print(frame.ix['a', 2]['Colorado'])  # 第一个[]表示行索引index，第二个[]表示列索引column
print(frame.ix['a', 2]['Ohio']['Red'])
print(frame.ix['a',2]['Ohio','Red'])
print('直接用MultiIndex创建层次索引结构')
print(MultiIndex.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Gree', 'Red', 'Green']],
                             names = ['state', 'color']))   # 这里索引的先后顺序会按照字母大小来排列

# reordering_and_sorting_levels 重新分级顺序
print('索引层级交换')
frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns = [['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
frame.index.names = ['key1', 'key2']
frame_swapped = frame.swaplevel('key1', 'key2')
print(frame_swapped)
print(frame_swapped.swaplevel(0, 1))
print('根据索引排序')
# print(frame.sortlevel('key2'))
# print(frame.swaplevel(0, 1).sortlevel(0))
print(frame.sort_index(level= 'key2'))   # 根据key2排序
print(frame.swaplevel(0, 1).sort_index(level = 0))  # 交换index 0和1,再对index为0的进行排序

# 根据级别汇总统计 summary_statistics_by_level
print('根据指定的key计算统计信息')
frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index = [['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns = [['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
frame.index.names = ['key1', 'key2']
print(frame)
print(frame.sum(level = 'key2'))

# 使用DataFrame的列 using_a_dataframes_columns
print('使用列生成层次索引')
frame = DataFrame({'a':range(7),
                   'b':range(7, 0, -1),
                   'c':['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd':[0, 1, 2, 0, 1, 2, 3]})
print(frame)
print(frame.set_index(['c', 'd']))  # 把c/d列变成索引
print(frame.set_index(['c', 'd'], drop = False)) # 列依然保留
frame2 = frame.set_index(['c', 'd'])
print(frame2.reset_index())

# 整数索引 integer_indexing
print('整数索引')
ser = Series(np.arange(3.))
print(ser)
try:
    print(ser[-1]) # 这里会有歧义,可能是索引中的-1,也可能是真实数据里面的倒数第一个元素
except:
    print(sys.exc_info()[0])
ser2 = Series(np.arange(3.), index = ['a', 'b', 'c'])
print(ser2[0])
print(ser2[-1])
print(ser2[-2])
ser3 = Series(range(3), index = [-5, 1, 3])
print(ser3.iloc[2])  # 避免直接用[2]产生的歧义
print('对DataFrame使用整数索引')
frame = DataFrame(np.arange(6).reshape((3, 2)), index = [2, 0, 1])
print(frame)
print(frame.iloc[0])  # 表示第一行
print(frame.iloc[:, 1])


