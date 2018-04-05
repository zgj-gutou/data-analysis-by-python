Excel中的空的cell读入pandas中是空值（NaN），这个NaN是个浮点类型，一般当作空值处理。
所以要先去除NaN在进行分隔字符串，因为 'float' object has no attribute 'split',也就是float类型NaN不能进行分割，否则会报错。
