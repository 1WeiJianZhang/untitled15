import pandas as pd
import numpy as np
s="4321431123212344331113321222442323112431"
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

#上面的转移矩阵为 df_2

df_1=str_count_df_p(s)
#print(df_1)
# 这是第四步
#print(np.array([0.1,0.3,0.2,0.4]).dot(np.linalg.matrix_power(np.array(df_1,dtype=np.float64),3)))#概率极限分布
def get(z):
    z = np.array(z, dtype=np.float64).T
    z= z - np.eye(3)
    z= np.append(z, np.array([1, 1, 1])).reshape(-1, 3)
    return np.linalg.lstsq(z,np.array([0,0,0,1]))
print(get([[0.2,0.8,0],
   [0.8,0,0.2],
   [0.1,0.3,0.6]]))
# 第一个返回值为我们要求得p1到p4
#(array([0.29393371, 0.28893058, 0.26829268, 0.14884303]),
 #array([1.02992113e-33]),
# 4,
 #array([2.01437748, 1.25270383, 1.05972215, 0.70596599]))
# 第二个返回值为残差。第三个返回值时秩，第四个为矩阵奇异值



