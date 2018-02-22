import numpy as np 
import matplotlib.pyplot as plt 
import random 

data[][2] = np.loadtxt('Lab3.txt')

np.random.shuffle(data)
cent1 = data[0]
cent2 = data[1]
print data

np.expand_dims(data[:,2], axis = 2)

data[0][3].append("color") 
print data
print cent1
print cent2
#plt.scatter(x = data[:,0], y = data[:,1], s = 2)
