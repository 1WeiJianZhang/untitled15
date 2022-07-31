from matplotlib import pyplot as plt
import numpy as np
from sklearn import ensemble
def huatu():
    time= []
    energy= []
    filename = 'data.txt'  # 如果没有文件，自动创建
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
                pass
            p_tmp, E_tmp = [float(i) for i in lines.split()]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            time.append(p_tmp)  # 添加新读取的数据
            energy.append(E_tmp)
            pass
    x = range(len(time))
    plt.figure(figsize=(10, 5))
    plt.plot(x, time)
    plt.plot(x, energy)
    plt.show()
    plt.cla()

def huatu1(data1,data2):
   #x = range(len(data1))
   plt.figure(figsize=(10, 5))
   #plt.scatter(data1[700: ,8], data1[700: ,9],marker='o')
   #plt.scatter(data2[:,0], data2[:,1], marker='^')
   for i in range(len(data1[1000: ,8])):
       plt.plot([data1[1000: ,8][i], data2[:,0][i]], [data1[1000: ,9][i], data2[:,1][i]], color='r')
       plt.scatter([data1[1000: ,8][i], data2[:,0][i]], [data1[1000: ,9][i], data2[:,1][i]], color='b')
   plt.show()

def huatu2(data1,data2,energy,time,xx,alpha):
    data3, data4 = [data1[0]], [data2[0]]
    for i in range(1, len(data1)):
        data3.append(min(data1[:i]))
        data4.append(min(data2[:i]))
    plt.subplot(2, 1, 1)
    plt.plot(data1,label="Cmax")
    plt.plot(data3, label="Best-Cmax",lw=2)
    plt.ylabel("Completion Time")
    plt.axhline(y=energy, c="r", ls="--", lw=2)
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.ylabel("Energy Consumption")
    plt.plot(data2,label="TEC")
    plt.plot(data4, label="Best-TEC",lw=2)
    plt.axhline(y=time, c="r", ls="--", lw=2)
    plt.xlabel("Episodes")
    plt.legend(loc='upper right')
    filename=f"./img/{xx}_{round(alpha,1)}_point_line.png"
    plt.savefig(filename)
    plt.show()

def plot_final_result(data1,name,alpha):
    data11 = [data1[0], data1[2], data1[4], data1[6]]
    data12 = [data1[1], data1[3], data1[5],data1[7]]
    data13 = [alpha * data1[0] + (1 - alpha) * data1[1], alpha * data1[2] + (1 - alpha) * data1[3],
              alpha * data1[4] + (1 - alpha) * data1[5], alpha * data1[6] + (1 - alpha) * data1[7]]
    size = 4
    x = np.arange(size)
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2

    plt.bar(x, data11, width=width, label='T-ration')
    plt.bar(x + width, data12, width=width, label='E-ration')
    plt.bar(x + 2 * width, data13, width=width, label='W-T-E-ration')
    plt.ylabel("Ration")
    labels = ["HA1", "HA2", "HA3", "DQN"]
    plt.xticks(x, labels, ma="center")
    plt.legend()
    filename = f"./img/{name}_{round(alpha,1)}_bar.png"
    plt.savefig(filename)
    plt.show()

def plot3(data,name,alpha):
    plt.scatter(data[:, 8],data[:, 9], c=range(len(data)), cmap=plt.cm.Reds)
    plt.xlabel("E-ration")
    plt.ylabel("T-ration")
    filename=f"./img/{name}_{round(alpha,1)}_scatter.png"
    plt.savefig(filename)
    plt.show()

def predict_obj(data,data2,xx,alpha):
    data_real1 = data[:, 8:9]
    data_real2 = data[:, 9]
    for i in range(len(data_real2)-1):
        for j in range(i,len(data_real2)):
            if (data_real1[i, 0] < data_real1[j, 0]) and (data_real2[j] < data_real2[j]):
                del data_real1[j,0]
                del data_real2[j]

    x_train, y_train = data_real1, data_real2

    #print(x_train)
    x_test1=[]
    #x_test, y_test = test[:, :2], test[:, 2]  # 测试时y没有噪声
    random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)  # 随机森林回归,并使用20个决策树
    random_forest_regressor.fit(x_train, y_train)  # 拟合模型
    #score = random_forest_regressor.score(x_train, y_train)
    if data2==0:
        for i in range(-1+(1000*(min(data[0:, 8]))).astype(np.int32), 1+(1000*(max(data[0:, 8]))).astype(np.int32)):
            x_test1.append([i * 0.001])
        result = random_forest_regressor.predict(x_test1)
        result = result.flatten()
        # print(result)
        plt.plot(x_test1, result)
        filename = f"./img/{xx}_{round(alpha,1)}_predict.png"
        plt.xlabel("E-ration")
        plt.ylabel("T-ration")
        plt.savefig(filename)
        plt.show()
    result1 = random_forest_regressor.predict([[data2]])
    result1 = result1.flatten()
    return result1[0]

import pandas as pd
import matplotlib.pyplot as plt

def ga_plot(all_history_Y,name,alpha):
    Y_history = pd.DataFrame(all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    filename = f"./img/{name}_{round(alpha,1)}_ga.png"
    plt.xlabel("Iterations")
    plt.ylabel("W-T-E-ration")
    plt.savefig(filename)
    plt.show()

def de_plot(all_history_Y,name,alpha):
    Y_history = pd.DataFrame(all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    filename = f"./img/{name}_{round(alpha,1)}_de.png"
    plt.xlabel("Iterations")
    plt.ylabel("W-T-E-ration")
    plt.savefig(filename)
    plt.show()

def sa_plot(all_history_Y,name,alpha):
    plt.plot(pd.DataFrame(all_history_Y).cummin(axis=0))
    filename = f"./img/{name}_{round(alpha,1)}_sa.png"
    plt.xlabel("Iterations")
    plt.ylabel("W-T-E-ration")
    plt.savefig(filename)
    plt.show()

def afsa_plot(all_history_Y,name,alpha):
    plt.plot(pd.DataFrame(all_history_Y).cummin(axis=0))
    filename = f"./img/{name}_{round(alpha,1)}_afsa.png"
    plt.xlabel("Iterations")
    plt.ylabel("W-T-E-ration")
    plt.savefig(filename)
    plt.show()

def compare_plot(all_history_Y, name, alpha):
    color1 = ['r', 'g', 'b', 'c', 'm', 'y']
    all_history_Y1=all_history_Y[:]
    col=np.argsort(all_history_Y1)
    plt.bar(range(len(all_history_Y)), all_history_Y, color=[color1[col[i]] for i in range(5)])
    plt.ylabel("W-T-E-Ration")
    labels = ["DQN", "GA", "SA", "DE", "AFSA"]
    plt.xticks(range(len(all_history_Y)), labels, ma="center")
    filename = f"./img/{name}_{round(alpha,1)}_compare.png"
    plt.savefig(filename)
    plt.show()
