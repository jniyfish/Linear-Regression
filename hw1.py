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
fun.linearRegression(x,y,1,20,0)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,20,0)
fun.Five_Fold(x,y,1,20,0)

print('------degree:5 dataSize:20 result:-------')
fun.linearRegression(x,y,5,20,0)
fun.LeaveOneOut(x,y,5,20,0)
fun.Five_Fold(x,y,5,20,0)

print('------degree:10 dataSize:20  result:------')
fun.linearRegression(x,y,10,20,0)
fun.LeaveOneOut(x,y,10,20,0)
fun.Five_Fold(x,y,10,20,0)

print('------degree:14 dataSize:20  result:------')
fun.linearRegression(x,y,14,20,0)
fun.LeaveOneOut(x,y,14,20,0)
fun.Five_Fold(x,y,14,20,0)

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')
print('')
print('')
#-----------c----------#

print("------partC model------")
plt.subplot(3,2,2)
plt.title('dataSize = 20,sin function',fontsize='10')

np.random.seed(500)
noise2=np.random.normal(0,0.4,20)
sinX = np.linspace(0,1,20)#
sinY = np.sin(sinX * math.pi*2)
sinYn = np.sin(sinX*math.pi*2) + noise2

#plt.plot(sinX,sinY,color='r',label="fff") #origin
#plt.plot(sinX,sinYn,color='blue',label="")#noise function

print("degree:1 dataSize:20")
fun.linearRegression(sinX,sinYn,1,20,0)
fun.Five_Fold(sinX,sinYn,1,20,0)
fun.LeaveOneOut(sinX,sinYn,1,20,0)
print("degree:5 dataSize:20")
fun.linearRegression(sinX,sinYn,5,20,0)
fun.Five_Fold(sinX,sinYn,5,20,0)
fun.LeaveOneOut(sinX,sinYn,5,20,0)
print("degree:10 dataSize:20")
fun.linearRegression(sinX,sinYn,10,20,0)
fun.Five_Fold(sinX,sinYn,10,20,0)
fun.LeaveOneOut(sinX,sinYn,10,20,0)
print("degree:14 dataSize:20")
fun.linearRegression(sinX,sinYn,14,20,0)
fun.Five_Fold(sinX,sinYn,14,20,0)
fun.LeaveOneOut(sinX,sinYn,14,20,0)

plt.legend(loc='best')


print('')
print('')


#######d-60######
x = np.linspace(0,1,60)
noise=np.random.normal(0,0.4,60)
y = np.sin(x * math.pi*2) + noise

plt.subplot(3,2,3)
plt.title('dataSize = 60',fontsize=10)

print('------degree:1 dataSize:60 result:------')
fun.linearRegression(x,y,1,60,0)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,60,0)
fun.Five_Fold(x,y,1,60,0)

print('------degree:5 dataSize:60 result:------')
fun.linearRegression(x,y,5,60,0)
fun.LeaveOneOut(x,y,5,60,0)
fun.Five_Fold(x,y,5,60,0)

print('------degree:10 dataSize:60 result:------')
fun.linearRegression(x,y,10,60,0)
fun.LeaveOneOut(x,y,10,60,0)
fun.Five_Fold(x,y,10,60,0)

print('------degree:14 dataSize:60 result:------')
fun.linearRegression(x,y,14,60,0)
fun.LeaveOneOut(x,y,14,60,0)
fun.Five_Fold(x,y,14,60,0)

print('')
print('')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')

#######d-160######
x = np.linspace(0,1,160)
noise=np.random.normal(0,0.4,160)
y = np.sin(x * math.pi*2) + noise
plt.subplot(3,2,4)
plt.title('dataSize = 160',fontsize=10)

print('------degree:1 dataSize:160 result:------')
fun.linearRegression(x,y,1,160,0)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,160,0)
fun.Five_Fold(x,y,1,160,0)

print('------degree:5 dataSize:160 result:------')
fun.linearRegression(x,y,5,160,0)
fun.LeaveOneOut(x,y,5,160,0)
fun.Five_Fold(x,y,5,160,0)

print('------degree:10 dataSize:160 result:------')
fun.linearRegression(x,y,10,160,0)
fun.LeaveOneOut(x,y,10,160,0)
fun.Five_Fold(x,y,10,160,0)

print('------degree:14 dataSize:160 result:------')
fun.linearRegression(x,y,14,160,0)
fun.LeaveOneOut(x,y,14,160,0)
fun.Five_Fold(x,y,14,160,0)

print('')
print('')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')

#######d-320######
x = np.linspace(0,1,320)
noise=np.random.normal(0,0.4,320)
y = np.sin(x * math.pi*2) + noise
plt.subplot(3,2,5)
plt.title('dataSize = 320',fontsize=10)

print('------degree:1 dataSize:320 result:------')
fun.linearRegression(x,y,1,320,0)#(x,y,degree,dataSize)
fun.LeaveOneOut(x,y,1,320,0)
fun.Five_Fold(x,y,1,320,0)

print('------degree:5 dataSize:320 result:------')
fun.linearRegression(x,y,5,320,0)
fun.LeaveOneOut(x,y,5,320,0)
fun.Five_Fold(x,y,5,320,0)

print('------degree:10 dataSize:320 result:------')
fun.linearRegression(x,y,10,320,0)
fun.LeaveOneOut(x,y,10,320,0)
fun.Five_Fold(x,y,10,20,0)

print('------degree:14 dataSize:320 result:------')
fun.linearRegression(x,y,14,320,0)
fun.LeaveOneOut(x,y,14,320,0)
fun.Five_Fold(x,y,14,320,0)

print('')
print('')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.legend(loc='best')


#--------partE-----------
plt.subplot(3,2,6)

plt.title('dataSize = 20,Regu sin function',fontsize='10')

np.random.seed(500)
noise2=np.random.normal(0,0.4,20)
sinX = np.linspace(0,1,20)#
sinY = np.sin(sinX * math.pi*2)
sinYn = np.sin(sinX*math.pi*2) + noise2

#plt.plot(sinX,sinYn,color='blue',label="Noise function")#noise function

lamba = 0
fun.linearRegression(sinX,sinYn,14,20,lamba)
fun.Five_Fold(sinX,sinYn,14,20,lamba)
fun.LeaveOneOut(sinX,sinYn,14,20,lamba)
print('')

lamba = 0.001/20
fun.linearRegression(sinX,sinYn,14,20,lamba)
fun.Five_Fold(sinX,sinYn,14,20,lamba)
fun.LeaveOneOut(sinX,sinYn,14,20,lamba)
print('')

lamba = 1/20
fun.linearRegression(sinX,sinYn,14,20,lamba)
fun.Five_Fold(sinX,sinYn,14,20,lamba)
fun.LeaveOneOut(sinX,sinYn,14,20,lamba)
print('')

print("----")
lamba = 50
fun.linearRegression(sinX,sinYn,14,20,lamba)
fun.Five_Fold(sinX,sinYn,14,20,lamba)
fun.LeaveOneOut(sinX,sinYn,14,20,lamba)
print('')


plt.legend(loc=3,fontsize=8,ncol=2)
plt.savefig('data_point')
plt.show()
