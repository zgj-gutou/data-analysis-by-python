import numpy as np
import pandas as pd

# 打开文件
fileNameStr = './朝阳医院2018年销售数据.xlsx'
xls = pd.ExcelFile(fileNameStr,dtype = 'object')  # dtype='object'的意思是会统一按照字符串类型读取数据，这样读取可以保证某些列的数值读取后是正确的
saleDf = xls.parse('Sheet1',dtype = 'object')
print(saleDf.head())  # 打印前5行看看
print(saleDf.shape)   # 查看行和列的数量
print(saleDf.dtypes)   # 查看每一列的数据类型

# 数据清洗
# 1如果要选择子集，可按如下语句
# subSaleDf = saleDf.loc[0:4,'购药时间':'销售数量']  # 即选择0到4行，购药时间，销售数量两列
# 2列名重命名
colNameDict = {'购药时间':'销售时间','社保卡号':'卡号'}   # 即把购药时间改为销售时间，社保卡号改为卡号（这个其实不用改，只是为了试试方法）
saleDf.rename(columns = colNameDict,inplace = True)
print(saleDf.head())
# 3缺失值处理（删除或其他处理）
print('删除缺失值之前的大小',saleDf.shape)
saleDf = saleDf.dropna(subset=['销售时间','卡号'],how = 'any')  # 删除销售时间，卡号两列中任何有缺失值的地方
print('删除缺失值之后的大小',saleDf.shape)
# 4数据类型转换astype，比如把字符串转换为数值,把字符串转换为日期数据类型
saleDf['销售数量'] = saleDf['销售数量'].astype('float')
saleDf['应收金额'] = saleDf['应收金额'].astype('float')
saleDf['实收金额'] = saleDf['实收金额'].astype('float')
print(saleDf['销售数量'].dtypes)  # 注意这里用的是dtype,下一样的DataFrame要用dtypes
print(saleDf.dtypes)
'''
定义函数：分割销售日期，获取销售日期
输入：timeColSer 销售时间这一列，是个Series数据类型
输出：分割后的时间，返回也是个Series数据类型
'''
def splitSaletime(timeColSer):
    timeList = []
    for value in timeColSer:
        # 例如2018-01-01 星期五，分割后为：2018-01-01
        dateStr = value.split(' ')[0]  # [0]是时间，[1]是一星期中的某天
        timeList.append(dateStr)
    # 将列表转换为一维数据Series类型
    timeSer = pd.Series(timeList)
    return timeSer
timeSer = saleDf.loc[:,'销售时间']   # 获取销售时间这一列
dateSer = splitSaletime(timeSer)   # 对销售时间进行分割，得到dateSer,注意此时的timeSer里面的时间仍旧是字符串类型
saleDf.loc[:,'销售时间'] = dateSer  # 把得到的新的销售时间赋值到原先的saleDf中
print(saleDf.head())
# 用to_datetime把字符串格式的日期改为日期格式，errors = 'coerce'表示如果原始数据不符合日期的格式，转换后的值为空值Nan
# format是原始数据中日期的格式
saleDf.loc[:,'销售时间'] = pd.to_datetime(saleDf.loc[:,'销售时间'],format = '%Y-%m-%d',errors = 'coerce')
# 上一条的转换过程中可能有不符合条件的日期导致有空值，所以要再做一次删除
saleDf.dropna(subset= ['销售时间','卡号'],how = 'any')
# 排序 ascending= True表示升序，false表示降序
saleDf = saleDf.sort_values(by = '销售时间',ascending= True)
saleDf = saleDf.reset_index(drop=True)  # 排序后行索引会变乱，需要修改成从0到N按顺序的索引值
print(saleDf.describe())  # 发现销售数量有小于0的异常值
# 下面要删除异常值，注意下面的语句操作
# 先查询
querySer = saleDf.loc[:,'销售数量']>0   # 找到销售数量>0的那些行
print('删除异常值前',saleDf.shape)
saleDf = saleDf.loc[querySer,:]  # 把它们填进去
print('删除异常值后',saleDf.shape)

# 4构建模型
'''
业务指标1:月均消费次数=总消费次数 / 月份数
总消费次数：同一天内，同一个人发生的所有消费算作一次消费
# drop_duplicates 根据列名（销售时间，社区卡号），如果这两个列值同时相同，只保留1条，将重复的数据删除
'''
kpi1_Df = saleDf.drop_duplicates(subset=['销售时间','卡号'])
totalI = kpi1_Df.shape[0]  # 0表示行数，即总消费次数
# 计算月份数，时间范围
kpi1_Df = kpi1_Df.sort_values(by = '销售时间',ascending = True)
kpi1_Df = kpi1_Df.reset_index(drop=True)  #重命名行名（index）
startTime= kpi1_Df.loc[0,'销售时间']  # 最小时间
endTime =kpi1_Df.loc[totalI-1,'销售时间']  # 最大时间
daysI = (endTime-startTime).days  # 计算天数
monthsI = daysI//30  # 月份数: 运算符“//”表示取整除  #返回商的整数部分，例如9//2 输出结果是4
kpi1_I = totalI//monthsI
print('业务指标1：月均消费次数=',kpi1_I)
# 指标2：月均消费金额 = 总消费金额 / 月份数
totalMoneyF=saleDf.loc[:,'实收金额'].sum()  #总消费金额
monthMoneyF=totalMoneyF / monthsI  #月均消费金额n
print('业务指标2：月均消费金额=',monthMoneyF)
# 指标3：客单价=总消费金额 / 总消费次数
'''
totalMoneyF：总消费金额
totalI：总消费次数
'''
pct=totalMoneyF / totalI
print('客单价：',pct)

