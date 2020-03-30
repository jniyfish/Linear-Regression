import numpy as np
import matplotlib.pyplot as plt
from random import sample

def NormalEquation(nA,testY,shape):
	nA = nA.reshape(shape,2)
	testY = testY.reshape(shape,1)
	theta = np.dot(nA.T,nA)
	if np.linalg.det(theta) == 0: #singular
		theta = np.dot(np.linalg.pinv(theta),np.dot(nA.T,testY)) 
	elif np.linalg.det(theta) !=0 : #nonsingular
		theta = np.dot(np.linalg.inv(theta),np.dot(nA.T,testY)) 
	return theta


def LeaveOneOut(x,y):
	nA = np.array([])
	errorSum = 0
	for i in range(0,20):
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
		theta = NormalEquation(nA,trainY,19)
			
		testError = testY - (theta[0]*testX + theta[1])
		testError = np.dot(testError.T,testError)/1
		errorSum = errorSum + testError
	errorSum = errorSum/20
	print("LOO error:	",errorSum)


def LinearRegression(x,y):

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
	theta = NormalEquation(nA,trainY,15)

	trainError = trainYe - (theta[0]*trainX + theta[1])
	trainError = np.dot(trainError.T,trainError)/15#error = (y-y')^2 summation
	print("train Error:	",trainError)

	testError = testY - (theta[0]*testX + theta[1]) # y^ = y-y'
	testError = np.dot(testError.T,testError) /5        #error = (y-y')^2 summation
	print("test Error:	",testError)
	plt.scatter(testX,testY,s=30,c='blue',marker='o',alpha=0.5,label='test(5)')
	plt.scatter(trainX,trainY,s=30,c='green',marker='x',alpha=0.5,label='train(15)')
	return theta

def Five_Fold(x,y):

#	nA = np.array([])
	errorSum = 0
	for i in range(0,5):
		testX = np.array([])
		trainX = np.array([])
		testY = np.array([])
		trainY = np.array([])
		nA = np.array([])
		for j in range(0,20):
			if (4*i)+4>j and j >=(4*i) :
				testX = np.append(testX,x[i])
				testY = np.append(testY,y[i])
			else :
				trainX = np.append(trainX,x[i])
				trainY = np.append(trainY,y[i])
				nA = np.append(nA,[x[i],1])

		theta = NormalEquation(nA,trainY,16)

		testError = testY - (theta[0]*testX + theta[1])
		testError = np.dot(testError.T,testError)/4  #each time has 4 MSE
		print(testError)
		errorSum = errorSum + testError

	errorSum = errorSum/5    #five fold
	print("5_Fold error:	",errorSum)

