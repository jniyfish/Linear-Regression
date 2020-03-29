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


testX=np.array([])
testY =np.array([])
trainX=np.array([])
trainY=np.array([])
nA = np.array([]) # for normal equtaion

for i in range(0,20): #spilt data 
	if i%4==0 :
		testX = np.append(testX,x[i])
		testY = np.append(testY,y[i])
	elif i%4!=0 :
		trainX = np.append(trainX,x[i])
		nA = np.append(nA,[x[i],1])
		trainY = np.append(trainY,y[i])

trainYe = trainY
nA = nA.reshape(15,2)                #use training data to slove normal equation
trainY = trainY.reshape(15,1)
theta = np.dot(nA.T,nA)
theta = np.dot(np.linalg.inv(theta),np.dot(nA.T,trainY)) 


trainError = trainYe - (theta[0]*trainX + theta[1])
trainError = np.dot(trainError.T,trainError)/15#error = (y-y')^2 summation
print("train Error:	",trainError)

testError = testY - (theta[0]*testX + theta[1]) # y^ = y-y'
testError = np.dot(testError.T,testError) /5        #error = (y-y')^2 summation
print("test Error:	",testError)
myLine = theta[0]*x + theta[1]

fun.LeaveOneOut(x,y)

plt.scatter(testX,testY,s=30,c='blue',marker='o',alpha=0.5,label='test(5)')
plt.scatter(trainX,trainY,s=30,c='green',marker='x',alpha=0.5,label='train(15)')

plt.xlabel('x軸',fontsize=8)
plt.ylabel('y軸',fontsize=8,rotation=0)
plt.plot(x,myLine,color = 'r',label = 'line of 5 fold')
plt.savefig('data_point')
plt.legend(loc='upper left')

#plt.show()



