import pandas as pd
import numpy as np
s="1110010011111110011110111111001111111110001101101111011011010111101110111101111110011011111100111"
#index上面的0表示当前的一步，就第一步，columns上面的0表示下一步
# 计算ab在s中出现的次数
def find_count(s,ab):
    ab_sum = 0
    for i in range(0,len(s)-1):
        if s[i:i+2] == ab:ab_sum+=1 # 计算‘ab’出现的个数
    return ab_sum

# 转移矩阵
def str_count_df(s):
    # 获得里面不重复的元素
    unique_items = np.unique(list(s))
    # 获得不重复元素个数
    n = unique_items.size
    # 默认行是这一次的，列是下一次的。类容是他们的转换情况
    df_ = pd.DataFrame(index=unique_items,columns=unique_items)
    for i in unique_items:
        for j in unique_items:
            df_.loc[i,j] = find_count(s,i+j)
    return df_

# 转移矩阵，概率
def str_count_df_p(s):
    # 获得里面不重复的元素
    unique_items = np.unique(list(s))
    # 获得不重复元素个数
    n = unique_items.size
    # 默认行是这一次的，列是下一次的。类容是他们的转换情况
    df_ = pd.DataFrame(index=unique_items,columns=unique_items)
    for i in unique_items:
        for j in unique_items:
            df_.loc[i,j] = find_count(s,i+j)
    df_ = df_.div(df_.sum(axis=1),axis='index')
    return df_
print(str_count_df_p(s))
print(find_count(s,'10'))


