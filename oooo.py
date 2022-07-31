import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mticker
from matplotlib.colors import LightSource
from matplotlib import cm
from matplotlib.ticker import FixedLocator
# setup the figure and axes
file=open(r'C:\Users\zwj\PycharmProjects\pythonProject25\250.csv')
COLOR = ["#FC6805", "#FFB628", "#65B017", "#99D8DB", "#9887BB"]
color_list = []
for i in range(9):
    for j in range(5):
        color_list.append(COLOR[j])

color_list = np.asarray(color_list)
color_flat=color_list.ravel()
file_data=pd.read_csv(file)
fig= plt.figure(figsize=(16, 12))
ax1 = fig.add_subplot(projection='3d')
top=file_data.values.ravel()
# fake data
_x = np.arange(0,5)
_y = np.arange(0,9)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()
print(x,y)
#top = np.random.random(45)
bottom = np.ones_like(top)
width = 1
depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, color=color_flat, shade=True)
ax1.set_title('N=250')
ax1.set_xlim(0,5)
ax1.set_xticklabels(("DQN","GA","SA","DE","AFSA"))
ax1.set_ylim(0,10)
plt.show()