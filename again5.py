#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-19
# Author: ZYunfei
# File func: draw func

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import os
myfont=FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')
sns.set(font=myfont.get_name())

class Painter:
    def __init__(self, load_csv, load_dir=None):
        if not load_csv:
            self.data = pd.DataFrame(columns=['episode reward','episode', 'Method'])
        else:
            self.load_dir = load_dir
            if os.path.exists(self.load_dir):
                print("==正在读取{}。".format(self.load_dir))
                self.data = pd.read_csv(self.load_dir).iloc[:,0:] # csv文件第一列是index，不用取。
                print("==读取完毕。")
            else:
                print("==不存在{}下的文件，Painter已经自动创建该csv。".format(self.load_dir))
                self.data = pd.DataFrame(columns=['episode reward', 'episode', 'Method'])
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.hue_order = None

    def setXlabel(self,label): self.xlabel = label

    def setYlabel(self, label): self.ylabel = label

    def setTitle(self, label): self.title = label

    def setHueOrder(self,order):
        """设置成['name1','name2'...]形式"""
        self.hue_order = order

    def addData(self, dataSeries, method, smooth = True):
        if smooth:
            dataSeries = self.smooth(dataSeries)
        size = len(dataSeries)
        for i in range(size):
            dataToAppend = {'episode reward':dataSeries[i],'episode':i+1,'Method':method}
            self.data = self.data.append(dataToAppend,ignore_index = True)

    def drawFigure(self):
        sns.set_theme(style="darkgrid")
        sns.set_style(rc={"linewidth": 1})
        print("==正在绘图...")
        sns.relplot(data = self.data, kind = "line", x = "episode", y = "episode reward",
                    hue= "Method", hue_order=None)
        plt.title(self.title,fontsize = 12)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        print("==绘图完毕！")
        plt.show()

    def smoothData(self, smooth_method_name,N):
        """对某个方法下的reward进行MA滤波，N为MA滤波阶数。"""
        begin_index = -1
        mode = -1  # mode为-1表示还没搜索到初始索引， mode为1表示正在搜索末尾索引。
        for i in range(len(self.data)):
            if self.data.iloc[i]['Method'] == smooth_method_name and mode == -1:
                begin_index = i
                mode = 1
                continue
            if mode == 1 and self.data.iloc[i]['episode'] == 1:
                self.data.iloc[begin_index:i,0] = self.smooth(
                    self.data.iloc[begin_index:i,0],N = N
                )
                print(self.data.iloc[begin_index:i,0])
                begin_index = -1
                mode = -1
                if self.data.iloc[i]['Method'] == smooth_method_name:
                    begin_index = i
                    mode = 1
            if mode == 1 and i == len(self.data) - 1:
                self.data.iloc[begin_index:,0]= self.smooth(
                    self.data.iloc[begin_index:,0], N=N
                )
        print("==对{}数据{}次平滑完成!".format(smooth_method_name,N))

    @staticmethod
    def smooth(data,N=10):
        n = (N - 1) // 2
        res = np.zeros(len(data))
        for i in range(len(data)):
            if i <= n - 1:
                res[i] = sum(data[0:2 * i+1]) / (2 * i + 1)
            elif i < len(data) - n:
                res[i] = sum(data[i - n:i + n +1]) / (2 * n + 1)
            else:
                temp = len(data) - i
                res[i] = sum(data[-temp * 2 + 1:]) / (2 * temp - 1)
        return res


data1=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\basic\\basic_15_2.0.csv')
data0=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\dueing\\dueing_15_2.0.csv')
data2=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\per\\per_15_2.0.csv')
data3=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\rain\\rain_15_3.0.csv')
data4=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\n_step\\n_step_15_2.0.csv')
data5=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\noisy\\noisy_15_2.0.csv')
data6=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\ddqn\\ddqn_15_2.0.csv')
#data7=pd.read_csv('C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\cdqn\\cdqn_45_2.0.csv')
#
list = data1.values.tolist()
list = np.array(list).flatten().tolist()
#
list1 = data0.values.tolist()
list1 = np.array(list1).flatten().tolist()
#
list2 = data2.values.tolist()
list2 = np.array(list2).flatten().tolist()

list3 = data3.values.tolist()
list3 = np.array(list3).flatten().tolist()
#
list4 = data4.values.tolist()
list4 = np.array(list4).flatten().tolist()

list5 = data5.values.tolist()
list5 = np.array(list5).flatten().tolist()

list6 = data6.values.tolist()
list6 = np.array(list6).flatten().tolist()

#list7 = data7.values.tolist()
#list7 = np.array(list7).flatten().tolist()


if __name__ == "__main__":
    painter = Painter(load_csv=True,load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\basic\\basic_20_2.0.csv')
    painter = Painter(load_csv=True, load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\dueing\\dueing_20_2.0.csv')
    painter = Painter(load_csv=True, load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\per\\per_20_2.0.csv')
    painter = Painter(load_csv=True, load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\rain\\rain_20_3.0.csv')
    painter = Painter(load_csv=True, load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\n_step\\n_step_20_2.0.csv')
    painter = Painter(load_csv=True, load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\noisy\\noisy_20_2.0.csv')
    painter = Painter(load_csv=True, load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\ddqn\\ddqn_20_2.0.csv')
    #painter = Painter(load_csv=True, load_dir='C:\\Users\\zwj\\PycharmProjects\\pythonProject25\\cdqn\\cdqn_75_2.0.csv')

    painter.smoothData('dueing', 11)
    painter.smoothData('basic',11)
    painter.smoothData('per', 11)
    painter.smoothData('rain', 11)
    painter.smoothData('n_step', 11)
    painter.smoothData('noisy', 11)
    painter.smoothData('ddqn', 11)
    #painter.smoothData('cdqn', 11)
    #basic
    painter.addData(list[0:1000],'basic')
    painter.addData(list[1000:2000], 'basic')
    painter.addData(list[2000:3000], 'basic')
    painter.addData(list[3000:4000], 'basic')
    painter.addData(list[4000:5000], 'basic')
    painter.addData(list[5000:6000], 'basic')
    painter.addData(list[6000:7000], 'basic')
    painter.addData(list[7000:8000], 'basic')
    painter.addData(list[8000:9000], 'basic')
    #dueing
    painter.addData(list1[0:1000], 'dueing')
    painter.addData(list1[1000:2000], 'dueing')
    painter.addData(list1[2000:3000], 'dueing')
    painter.addData(list1[3000:4000], 'dueing')
    painter.addData(list1[4000:5000], 'dueing')
    painter.addData(list1[5000:6000], 'dueing')
    painter.addData(list1[6000:7000], 'dueing')
    painter.addData(list1[7000:8000], 'dueing')
    painter.addData(list1[8000:9000], 'dueing')
    #per
    painter.addData(list2[0:1000], 'per')
    painter.addData(list2[1000:2000], 'per')
    painter.addData(list2[2000:3000], 'per')
    painter.addData(list2[3000:4000], 'per')
    painter.addData(list2[4000:5000], 'per')
    painter.addData(list2[5000:6000], 'per')
    painter.addData(list2[6000:7000], 'per')
    painter.addData(list2[7000:8000], 'per')
    painter.addData(list2[8000:9000], 'per')
    #rain
    painter.addData(list3[0:1000], 'rain')
    painter.addData(list3[1000:2000], 'rain')
    painter.addData(list3[2000:3000], 'rain')
    painter.addData(list3[3000:4000], 'rain')
    painter.addData(list3[4000:5000], 'rain')
    painter.addData(list3[5000:6000], 'rain')
    painter.addData(list3[6000:7000], 'rain')
    painter.addData(list3[7000:8000], 'rain')
    painter.addData(list3[8000:9000], 'rain')
    # n_step
    painter.addData(list4[0:1000], 'n_step')
    painter.addData(list4[1000:2000], 'n_step')
    painter.addData(list4[2000:3000], 'n_step')
    painter.addData(list4[3000:4000], 'n_step')
    painter.addData(list4[4000:5000], 'n_step')
    painter.addData(list4[5000:6000], 'n_step')
    painter.addData(list4[6000:7000], 'n_step')
    painter.addData(list4[7000:8000], 'n_step')
    painter.addData(list4[8000:9000], 'n_step')
    #noisy
    painter.addData(list5[0:1000], 'noisy')
    painter.addData(list5[1000:2000], 'noisy')
    painter.addData(list5[2000:3000], 'noisy')
    painter.addData(list5[3000:4000], 'noisy')
    painter.addData(list5[4000:5000], 'noisy')
    painter.addData(list5[5000:6000], 'noisy')
    painter.addData(list5[6000:7000], 'noisy')
    painter.addData(list5[7000:8000], 'noisy')
    painter.addData(list5[8000:9000], 'noisy')
    #ddqn
    painter.addData(list6[0:1000], 'ddqn')
    painter.addData(list6[1000:2000], 'ddqn')
    painter.addData(list6[2000:3000], 'ddqn')
    painter.addData(list6[3000:4000], 'ddqn')
    painter.addData(list6[4000:5000], 'ddqn')
    painter.addData(list6[5000:6000], 'ddqn')
    painter.addData(list6[6000:7000], 'ddqn')
    painter.addData(list6[7000:8000], 'ddqn')
    painter.addData(list6[8000:9000], 'ddqn')
    # cdqn
    #painter.addData(list7[0:1000], 'cdqn')
    #painter.addData(list7[1000:2000], 'cdqn')
    #painter.addData(list7[2000:3000], 'cdqn')
    #painter.addData(list7[3000:4000], 'cdqn')
    #painter.addData(list7[4000:5000], 'cdqn')
    #painter.addData(list7[5000:6000], 'cdqn')
    #painter.addData(list7[6000:7000], 'cdqn')
    #painter.addData(list7[7000:8000], 'cdqn')
    #painter.addData(list7[8000:9000], 'cdqn')

    painter.drawFigure()

