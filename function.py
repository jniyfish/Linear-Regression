import numpy as np
import matplotlib.pyplot as plt
import math
from random import sample

def NormalEquation(nA,testY,shape_m,shape_n):  #X:mxn matrix
	nA = nA.reshape(shape_m,shape_n)
	testY = testY.reshape(shape_m,1)
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
		theta = NormalEquation(nA,trainY,19,2)
			
		testError = testY - (theta[0]*testX + theta[1])
		testError = np.dot(testError.T,testError)/1
		errorSum = errorSum + testError
	errorSum = errorSum/20
	print("LOO error:	",errorSum)


def Five_Fold(x,y):
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

		theta = NormalEquation(nA,trainY,16,2)

		testError = testY - (theta[0]*testX + theta[1])
		testError = np.dot(testError.T,testError)/4  #each time has 4 MSE
		errorSum = errorSum + testError

	errorSum = errorSum/5    #five fold
	print("5_Fold error:	",errorSum)


def PolyReression(x,y,degree):

	testX=np.array([])
	testY =np.array([])
	trainX=np.array([])
	trainY=np.array([])
	nA = np.array([]) # for normal equtaion
	plotX = np.array([])
	for i in range(0,20): #spilt data 
		row = np.array([])
		if i%4==0 :
			for j in range(degree,0,-1):
				row  = np.append(row,math.pow(x[i],j))
			row = np.append(row,1)
			testX = np.append(testX,x[i])
			testY = np.append(testY,y[i])
		elif i%4!=0 :
			for j in range(degree,0,-1):
				row = np.append(row,math.pow(x[i],j))
			row = np.append(row,1)
			nA = np.append(nA,row)
			plotX = np.append(plotX,x[i])
			trainY = np.append(trainY,y[i])
		
	theta = NormalEquation(nA,trainY,15,degree+1)
	fitY = np.zeros(15)
	fitTestY = np.zeros(5)
	j = degree

	for i in range(0,degree):
		fitY = fitY + theta[i]*(np.power(plotX,j))
		fitTestY = fitTestY + theta[i]*(np.power(testX,j))
		j = j - 1
	fitY = fitY + theta[degree]
	fitTestY = fitTestY + theta[degree]

	trainError = trainY - fitY
	trainError =  np.dot(trainError.T,trainError)/15
	print('Degree ',degree,' train Error ',trainError)
	testError = testY - fitTestY
	testError = np.dot(testError.T,testError)/5
	print('Degree ',degree,'test Error	',testError)
	print('')
	if degree == 1:
		plt.plot(plotX,fitY,color='r',label='1-degree')
	elif degree == 5:
		plt.plot(plotX,fitY,color='g',label='5-degree')
	elif degree ==10:
		plt.plot(plotX,fitY,color='y',label='10-degree')
	elif degree ==14:	
		plt.plot(plotX,fitY,color='pink',label='14-degree')
