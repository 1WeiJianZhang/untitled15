import numpy as np
from numpy.random import randint
import math

points = [(1, 2), (3, 6), (5, 8), (4, 9), (7, 6),
          (7, 3), (4, 2), (9, 0), (7, 8), (1, 6), (1, 2)]
distances = []
x = [0, 1, 3, 5, 6, 2, 4, 7, 8, 9, 10]
for i in points:
    distances_temp = []
    for j in points:
        temp = (i[0] - j[0]) * (i[0] - j[0]) + (i[1] - j[1]) * (i[1] - j[1])
        temp = np.sqrt(temp)
        distances_temp.append(temp)
    distances.append(distances_temp)

def cal_distance(x, distances):
    temp = 0
    for i in range(len(x) - 1):
        temp += distances[x[i]][x[i + 1]]
    return temp

def shake(x, k, distances):
    temp_res = 0
    x1 = x[:]
    if k == 1:
        c = randint(1, 10, 2)
        c.sort()
        temp = x1[c[0]:c[1]]
        i = 0
        while temp:
            x1[c[0] + i] = temp.pop()
            i = i + 1
        temp_res = cal_distance(x1, distances)
    else:
        temp_res = 0
        c = randint(1, 10, 2)
        c.sort()
        temp = x1[c[0]]
        x1[c[0]] = x1[c[1]]
        x1[c[1]] = temp
        temp_res = cal_distance(x1, distances)

    return x1, temp_res


def localsearch(x, k, distances):
    temp_res = cal_distance(x, distances)
    x2 = []
    if k == 1:
        for i in range(1, len(x)-1):
            for j in range(i+1, len(x)):
                ii = 0
                temp = x[i:j]
                x1 = x[:]
                while temp:
                    x1[i + ii] = temp.pop()
                    ii=ii+1
                if temp_res > cal_distance(x1, distances):
                    temp_res = cal_distance(x1, distances)
                    x2 = x1[:]
    else:
        for i in range(1, len(x)):
            for j in range(i, len(x)):
                x1 = x[:]
                temp = x1[i]
                x1[i] = x1[j]
                x1[j] = temp
                if temp_res > cal_distance(x1, distances):
                    temp_res = cal_distance(x1, distances)
                    x2 = x1[:]
    return x2, temp_res

while True:
    k = 1
    temp_best=cal_distance(x, distances)
    while k < 3:
        x1,temp_res = shake(x,k,distances)
        if temp_res <temp_best:
            temp_best=temp_res
        x2,remp_res = localsearch(x1,k,distances)
        if temp_res <temp_best:
            temp_best=temp_res
        print(temp_best)






