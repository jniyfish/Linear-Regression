import numpy as np
import matplotlib.pyplot as plt
from random import sample
import function as fun

np.random.seed(0)
noise=np.random.normal(0,1,20)
#rng = np.random.RandomState(1)
#noise = rng.randn(20)
x = np.linspace(3,-3,20)
y=2*x+noise



print('partA')
fun.PolyReression(x,y,1)
fun.LeaveOneOut(x,y)
fun.Five_Fold(x,y)
print('partB')
fun.PolyReression(x,y,5)
fun.PolyReression(x,y,10)
fun.PolyReression(x,y,14)



plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.savefig('data_point')
plt.legend(loc='upper left')

#plt.show()



