import numpy as np
import pandas as pd
from pandas import Series, DataFrame

ser = Series(np.arange(3.))
data = DataFrame(np.arange(16).reshape(4,4),index=list('abcd'),columns=list('wxyz'))

# [ ]切片
# 列选择，根据列索引，注意不能进行某个范围的多列选择比如data[['w':'z']],而只能指定某几列的多列选择data[['w','z']]
print( data['w'] )  #选择表格中的'w'列，使用类字典属性,返回的是Series类型
print( data.w )    #选择表格中的'w'列，使用点属性,返回的是Series类型
print( data[['w']] )  # 选择表格中的'w'列，返回的是DataFrame属性,可以将输出与上面两句语句对比一下！
print( data[['w','z']] )  # 选择表格中的'w'、'z'列,返回的是DataFrame属性
# 行选择，根据行号（索引的位置）
print( data[0:2] )  # 返回第1行到第2行的所有行，前闭后开，包括前不包括后,返回的是DataFrame
print( data[1:2] )  # 返回第2行，从0计，返回的是单行,返回的是DataFrame，通过有前后值的索引形式，   # 如果采用data[1]则报错
print( data.ix[1:2] ) # 返回第2行的第三种方法，返回的是DataFrame，跟data[1:2]同
# 行选择，根据行索引index
print( data['a':'b'] )  # 利用index值进行切片，返回的是**前闭后闭**的DataFrame,   #即末端是包含的
# 区块选择
print( data[:2][['w','z']])  # 区块选择就是把行选择和列选择结合起来。注意和行选择和列选择的比较

# loc切片
# 行选择,根据行索引index,注意这里的行索引不能换成行号，除非建dataframe时没有指定行索引只有默认的行索引，也就是数字
print( data.loc['a':'b'] )  # 返回的是第1行到第3行的所有行，返回的是前闭后闭的DataFrame，和上面的行选择不同。
# 列选择,根据列索引选择。这里的写法有点像区域选择。
print( data.loc[:,['w','z']] )   # 返回的是DataFrame，但是data.loc[:,['w':'z']]写法是错误的，也就是只能指定某几列，而不是列的范围
# 区块选择
print( data.loc['a':'b',['w','z']] )  # 返回的是DataFrame
print( data.loc[['a','b'],['w','z']] )   # 返回的是DataFrame。但是data.loc[['a':'b'],['w','z']]写法是错的
print( data.loc['a':'b',['w']] )   # 返回的是DataFrame
print( data.loc['a':'b','w'] )    # 返回的是Series

# iloc切片
# 行选择
print( data.iloc[0:2] )  # 返回的是第1行到第2行的所有行，返回的是前闭后开的DataFrame
print( data.iloc[[0,2]] )  # 返回的是第1行和第3行的所有行，返回的是DataFrame
print( data.iloc[0,2] )   # 返回的是第1行和第3列的那个数字
# 列选择
print( data.iloc[:,[0,2]] )  # 返回的是第1列到第2列的所有行，返回的是前闭后开的DataFrame，注意要有冒号代表所有行
# 区块选择
print( data.iloc[0:2,[0,2]])  # 返回的是第1行到第2行，第1列和第2列的DataFrame
print( data.iloc[[0,2],[0,2]])  # 返回的是第1行和第3行，第1列和第2列的DataFrame


