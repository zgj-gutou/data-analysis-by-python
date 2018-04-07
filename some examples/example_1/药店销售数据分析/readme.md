Excel中的空的cell读入pandas中是空值（NaN），这个NaN是个浮点类型，一般当作空值处理。<br>
所以要先去除NaN再进行分隔字符串，因为 'float' object has no attribute 'split',也就是float类型NaN不能进行分割，否则会报错。<br>
那None和NaN有什么区别呢：None是Python的一种数据类型，NaN是浮点类型，两个都用作空值
```python
#None和NaN的区别
print('None的数据类型',type(None))
from numpy import NaN
print('NaN的数据类型',type(NaN))
```
None的数据类型 <class 'NoneType'> <br>
NaN的数据类型 <class 'float'>


