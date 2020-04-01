import numpy as np
import matplotlib.pyplot as plt
from random import sample
import function as fun
import math

np.random.seed(500)
noise=np.random.normal(0,1,20)
#rng = np.random.RandomState(1)
#noise = rng.randn(20)
x = np.linspace(3,-3,20)
y=2*x+noise
plt.subplot(3,2,1)
plt.plot([0,1],[0,1])
plt.title('dataSize = 20',fontsize=10)


print('------degree:1  dataSize:20  result:------')
fun.linearRegression(x,y,1,20)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,20)
fun.Five_Fold(x,y,1,20)

print('------degree:5 dataSize:20 result:-------')
fun.linearRegression(x,y,5,20)
fun.LeaveOneOut(x,y,5,20)
fun.Five_Fold(x,y,5,20)

print('------degree:10 dataSize:20  result:------')
fun.linearRegression(x,y,10,20)
fun.LeaveOneOut(x,y,10,20)
fun.Five_Fold(x,y,10,20)

print('------degree:14 dataSize:20  result:------')
fun.linearRegression(x,y,14,20)
fun.LeaveOneOut(x,y,14,20)
fun.Five_Fold(x,y,14,20)

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')
print('')
print('')
########c########

print("------partC model------")
plt.subplot(3,2,2)
plt.title('dataSize = 20,sin function',fontsize='10')

np.random.seed(500)
noise2=np.random.normal(0,0.4,20)
sinX = np.linspace(3,-3,20)#
sinY = np.sin(sinX * math.pi*2)
sinYn = np.sin(sinX*math.pi*2) + noise2

plt.plot(sinX,sinY,color='r',label="fff") #origin
plt.plot(sinX,sinYn,color='blue',label="fff")#noise function

print("degree:1 dataSize:20")
fun.linearRegression(sinX,sinY,1,20)
print("degree:5 dataSize:20")
fun.linearRegression(sinX,sinY,5,20)
print("degree:10 dataSize:20")
fun.linearRegression(sinX,sinY,10,20)
print("degree:14 dataSize:20")
fun.linearRegression(sinX,sinY,14,20)



print('')
print('')


#######d-60######
x = np.linspace(3,-3,60)
noise=np.random.normal(0,1,60)
y=2*x+noise
plt.subplot(3,2,3)
plt.title('dataSize = 60',fontsize=10)

print('------degree:1 dataSize:60 result:------')
fun.linearRegression(x,y,1,60)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,60)
fun.Five_Fold(x,y,1,60)

print('------degree:5 dataSize:60 result:------')
fun.linearRegression(x,y,5,60)
fun.LeaveOneOut(x,y,5,60)
fun.Five_Fold(x,y,5,60)

print('------degree:10 dataSize:60 result:------')
fun.linearRegression(x,y,10,60)
fun.LeaveOneOut(x,y,10,60)
fun.Five_Fold(x,y,10,60)

print('------degree:14 dataSize:60 result:------')
fun.linearRegression(x,y,14,60)
fun.LeaveOneOut(x,y,14,60)
fun.Five_Fold(x,y,14,60)

print('')
print('')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')

#######d-160######
x = np.linspace(3,-3,160)
noise=np.random.normal(0,1,160)
y=2*x+noise
plt.subplot(3,2,4)
plt.title('dataSize = 160',fontsize=10)

print('------degree:1 dataSize:160 result:------')
fun.linearRegression(x,y,1,160)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,160)
fun.Five_Fold(x,y,1,160)

print('------degree:5 dataSize:160 result:------')
fun.linearRegression(x,y,5,160)
fun.LeaveOneOut(x,y,5,160)
fun.Five_Fold(x,y,5,160)

print('------degree:10 dataSize:160 result:------')
fun.linearRegression(x,y,10,160)
fun.LeaveOneOut(x,y,10,160)
fun.Five_Fold(x,y,10,160)

print('------degree:14 dataSize:160 result:------')
fun.linearRegression(x,y,14,160)
fun.LeaveOneOut(x,y,14,160)
fun.Five_Fold(x,y,14,160)

print('')
print('')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')

#######d-320######
x = np.linspace(3,-3,320)
noise=np.random.normal(0,1,320)
y=2*x+noise
plt.subplot(3,2,5)
plt.title('dataSize = 320',fontsize=10)

print('------degree:1 dataSize:320 result:------')
fun.linearRegression(x,y,1,320)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,320)
fun.Five_Fold(x,y,1,320)

print('------degree:5 dataSize:320 result:------')
fun.linearRegression(x,y,5,320)
fun.LeaveOneOut(x,y,5,320)
fun.Five_Fold(x,y,5,320)

print('------degree:10 dataSize:320 result:------')
fun.linearRegression(x,y,10,320)
fun.LeaveOneOut(x,y,10,320)
fun.Five_Fold(x,y,10,20)

print('------degree:14 dataSize:320 result:------')
fun.linearRegression(x,y,14,320)
fun.LeaveOneOut(x,y,14,320)
fun.Five_Fold(x,y,14,320)

print('')
print('')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')


plt.savefig('data_point')
plt.show()
