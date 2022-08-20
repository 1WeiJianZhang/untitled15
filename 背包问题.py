#j<w(i)      V(i,j)=V(i-1,j)   （装不下i，因此不考虑）
#j>=w(i)     V(i,j)=max｛V(i-1,j)，V(i-1,j-w(i))+v(i)｝（装与不装选一个，装的时候的表达式）
w=[0,2,3,4,5]
v=[0,3,4,5,6]
bagv=8
dp=[[0 for i in range(9)]for i in range(5)]
#print(dp)
def findMax1():
    for i in range(5):
        for j in range(bagv + 1):
            if j < w[i]:
                dp[i][j] = dp[i - 1][j];
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);

def print2():
    for i in range(5):
        res = []
        for j in range(9):
            res.append(dp[i][j])
        print(res)
findMax1()
print2()




##找出被选中的商品
w=[0,2,3,4,5]
v=[0,3,4,5,6]
bagv=8
dp=[[0 for i in range(9)]for i in range(5)]
item = [0]*5
#print(item)
def findMax(): #动态规划
    for i in range (5):
        for j in range(bagv+1):
            if j < w[i]:
                dp[i][j] = dp[i - 1][j];
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w[i]] + v[i]);
def findWhat(i,j): #最优解情况
    if i>=0:
        if dp[i][j] == dp[i-1][j]:
            item[i] = 0
            findWhat(i-1,j)
        elif j - w[i] >= 0  and  dp[i][j] == dp[i - 1][j - w[i]] + v[i]:
            item[i] = 1
            findWhat(i-1,j-w[i])

def print1():
    for i in range(5):
        res = []
        for j in range(9):
            res.append(dp[i][j])
        #print(res)
    #print(item)
findMax()
findWhat(4,8)
print1()


