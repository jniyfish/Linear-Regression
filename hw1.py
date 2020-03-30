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
plt.subplot(2,1,1)
plt.plot([0,1],[0,1])
plt.scatter(x,y,s=30,c='blue',marker='o',alpha=0.5,label='test(5)')


print('degree1 result:')
fun.linearRegression(x,y,1)
fun.LeaveOneOut(x,y,1)
fun.Five_Fold(x,y,1)
print('------------')

print('degree5 result:')
fun.linearRegression(x,y,5)
fun.LeaveOneOut(x,y,5)
fun.Five_Fold(x,y,5)
print('------------')

print('degree10 result:')
fun.linearRegression(x,y,10)
fun.LeaveOneOut(x,y,10)
fun.Five_Fold(x,y,10)
print('------------')

print('degree14 result:')
fun.linearRegression(x,y,14)
fun.LeaveOneOut(x,y,14)
fun.Five_Fold(x,y,14)
print('------------')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.savefig('data_point')
plt.legend(loc='upper left')

########c.d########
plt.subplot(2,1,2)
np.random.seed(500)
noise2=np.random.normal(0,0.4,20)
sinX = np.linspace(3,-3,20)
sinY = np.sin(sinX * math.pi*2)
sinYn = np.sin(sinX*math.pi*2) + noise2

plt.plot(sinX,sinY,color='r',label="fff")
plt.plot(sinX,sinYn,color='blue',label="fff")


fun.linearRegression(sinX,sinY,1)
fun.linearRegression(sinX,sinY,5)
fun.linearRegression(sinX,sinY,10)
fun.linearRegression(sinX,sinY,14)

plt.show()



