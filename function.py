import numpy as np
import matplotlib.pyplot as plt
from random import sample

def LeaveOneOut(x,y):
	nA = np.array([])
	errorSum = 0
	for i in range(1,20):
		testX = np.array([])
		trainX = np.array([])
		testY = np.array([])
		trainY = np.array([])
		nA = np.array([])
		for j in range(0,20):
			if j==i:
				testX = np.append(testX,x[i])
				testY = np.append(testY,y[i])
			elif j!=i:
				trainX = np.append(trainX,x[i])
				trainY = np.append(trainY,y[i])
				nA = np.append(nA,[x[i],1])

		trainYe = trainY
		nA = nA.reshape(19,2)
		trainY = trainY.reshape(19,1)
		theta = np.dot(nA.T,nA)
		if np.linalg.det(theta) == 0:
			theta = np.dot(np.linalg.pinv(theta),np.dot(nA.T,trainY)) 
		elif np.linalg.det(theta) !=0 :
			theta = np.dot(np.linalg.inv(theta),np.dot(nA.T,trainY)) 
		testError = testY - (theta[0]*testX + theta[1])
		testError = np.dot(testError.T,testError)/1
		errorSum = errorSum + testError
	errorSum = errorSum/20
	print(errorSum)
