# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:01:18 2023

@author: HP
"""

import pandas as pd
import sklearn
import matplotlib.pyplot as plt 

main = pd.read_excel('C:/Users/HP/PycharmProjects/ezinne_explor_anlsis/main.xlsx','House price growth')
main2 = pd.read_excel('C:/Users/HP/PycharmProjects/ezinne_explor_anlsis/main.xlsx','Average pay')
main3 = pd.read_excel('C:/Users/HP/PycharmProjects/ezinne_explor_anlsis/main.xlsx')

# inflation
y2 =main3['CPIH']
x2 =main3['period']

#house growth
y =main['House price growth']
x =main['time']

#average pay
x1 =main2['Period']
y1 =main2['Regular pay (real)']

plt.figure(1)
plt.bar(x1,y1)

plt.figure(2)
plt.plot(x,y)

plt.figure(3)
plt.plot(x2,y2)
plt.show()