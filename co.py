import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import IndexLocator
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['font.size']=2


plot_data=np.random.rand(4,6)
plot_data=plot_data*100
plot_data=pd.DataFrame(plot_data,columns=["S0","S1","S2",'S3','S4','S5'])

fig_width = 9
fig_height = 6
width = fig_width / 2.54
height = fig_height / 2.54
dpi = 600
#fig, ax = plt.subplots(1, 1, figsize=(width, height), dpi=100)
fig = plt.figure(figsize=(width, height), dpi=dpi)
ax = fig.add_subplot(111,projection='3d')
fig.subplots_adjust(0.0, 0.00, 0.80, 1)


row_list=list(plot_data.index)
for z in row_list:
	ax.bar(plot_data.columns,plot_data.loc[z],zs=row_list.index(z),zdir='y',alpha=0.7)

ax.set_xlim(-0.9,5.9)
ax.set_zlim(-0.3,3.4)
ax.set_zlim(0,90)
#这里是使用了locator这个定位器来设定轴刻度，可以参考我的博文：
#https://blog.csdn.net/yue81560/article/details/107729622
ax.zaxis.set_major_locator(IndexLocator(20,0))
ax.set_yticks(range(len(row_list)))
plt.setp(ax.get_xticklabels(), visible=False)
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_zticklabels(), visible=False)
ax.set_xlim(-0.9,5.9)
ax.set_zlim(-0.3,3.4)
ax.set_zlim(0,90)
for yticklabel in row_list:
	ax.text(6.5, row_list.index(yticklabel) - 0.15, -1, yticklabel)

xticklabels = list(plot_data.columns)

for xticklabel in xticklabels:
	ax.text(xticklabels.index(xticklabel), -0.7, 0, xticklabel)

for zticklabel in range(0, 100, 20):
	if zticklabel == 0:
		ax.text(6.0, 3.6, zticklabel - 8, zticklabel)
	else:
		ax.text(6.0, 3.6, zticklabel - 10, '{0}%'.format(zticklabel))

plt.show()