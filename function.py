import numpy as np
import matplotlib.pyplot as plt
import math
from random import sample

def NormalEquation(nA,testY,shape_m,shape_n,lamba):  #nA = X:mxn matrix
	nA = nA.reshape(shape_m,shape_n)
	testY = testY.reshape(shape_m,1)
	if lamba ==0: #(X^T * X) 
		theta = np.dot(nA.T,nA)
	elif lamba != 0:  #(x^T*X )minus lamba * I
		theta = np.dot(nA.T,nA)
		theta = theta -(lamba * np.identity(shape_n))
	if np.linalg.det(theta) == 0: #singular
		theta = np.dot(np.linalg.pinv(theta),np.dot(nA.T,testY)) 
	elif np.linalg.det(theta) !=0 : #nonsingular
		theta = np.dot(np.linalg.inv(theta),np.dot(nA.T,testY)) 
	return theta



def LeaveOneOut(x,y,degree,dataSize,lamba):
	trainSize = dataSize - 1
	testSize = 1   #only one test
	errorSum = 0
	for i in range(0,dataSize):
		testX = np.array([])
		trainX = np.array([])
		testY = np.array([])
		trainY = np.array([])
		nA = np.array([])
		plotX = np.array([])
		for j in range(0,dataSize):
			row = np.array([])
			if j==i:
				testX = np.append(testX,x[i])
				testY = np.append(testY,y[i])
			elif j!=i:
				for j in range(degree,0,-1):
					row = np.append(row,math.pow(x[i],j))
				row = np.append(row,1)
				plotX = np.append(plotX,x[i])
				trainX = np.append(trainX,x[i])
				trainY = np.append(trainY,y[i])
				nA = np.append(nA,row)
		result = Reression(nA,plotX,testX,testY,trainX,trainY,degree,trainSize,testSize,lamba)
		errorSum = errorSum + result[0]
	errorSum = errorSum/20
	print("LOO error:	",errorSum)


def Five_Fold(x,y,degree,dataSize,lamba):
	trainSize = dataSize*4//5  #each time use 4/5 for train 
	testSize = dataSize//5     # 1/5 for test
	errorSum = 0
	for i in range(0,5):
		testX = np.array([])
		trainX = np.array([])
		testY = np.array([])
		trainY = np.array([])
		plotX=np.array([])
		nA = np.array([])
		for j in range(0,dataSize):
			row = np.array([])
			if (testSize*i)+testSize>j and j >=(testSize*i) :
				testX = np.append(testX,x[i])
				testY = np.append(testY,y[i])
			else :
				for j in range(degree,0,-1):
					row = np.append(row,math.pow(x[i],j))
				row = np.append(row,1)
				plotX = np.append(plotX,x[i])
				trainX = np.append(trainX,x[i])
				trainY = np.append(trainY,y[i])
				nA = np.append(nA,row)

		result = Reression(nA,plotX,testX,testY,trainX,trainY,degree,trainSize,testSize,lamba)
		errorSum = errorSum + result[0]

	errorSum = errorSum/5    #five fold
	print("5_Fold error:	",errorSum)

def Reression(nA,plotX,testX,testY,trainX,trainY,degree,trainSize,testSize,lamba):

	theta = NormalEquation(nA,trainY,trainSize,degree+1,lamba)
	fitY = np.zeros(trainSize)
	fitTestY = np.zeros(testSize)
	j = degree

	for i in range(0,degree):
		fitY = fitY + theta[i]*(np.power(plotX,j))
		fitTestY = fitTestY + theta[i]*(np.power(testX,j))
		j = j - 1
	fitY = fitY + theta[degree]
	fitTestY = fitTestY + theta[degree]
	trainError = trainY - fitY  
	trainError =  np.dot(trainError.T,trainError)/trainSize
	testError = testY - fitTestY
	testError = np.dot(testError.T,testError)/testSize
	
	return testError,trainError,plotX,fitY
def linearRegression(x,y,degree,dataSize,lamba):

	testSize = dataSize//4
	trainSize = dataSize*3//4
	testX=np.array([])
	testY =np.array([])
	trainX=np.array([])
	trainY=np.array([])
	nA = np.array([]) # for normal equtaion
	plotX = np.array([])
	for i in range(0,dataSize): #spilt data 
		row = np.array([])
		if i%4==2 :  #1/4 for test
			testX = np.append(testX,x[i])
			testY = np.append(testY,y[i])
		elif i%4!=2 :#3/4 for train
			for j in range(degree,0,-1):
				row = np.append(row,math.pow(x[i],j))
			row = np.append(row,1)
			nA = np.append(nA,row)
			plotX = np.append(plotX,x[i])
			trainY = np.append(trainY,y[i])
			
	result = Reression(nA,plotX,testX,testY,trainX,trainY,degree,trainSize,testSize,lamba)
	print('train Error ',result[1])
	print('test Error	',result[0])
	if lamba == 0:
		if degree == 1:
			plt.scatter(plotX,trainY,s=40,c='blue',marker='o',alpha=0.5,label='train')
			plt.scatter(testX,testY,s=40,c='green',marker='x',alpha=0.5,label='test')
			plt.plot(plotX,result[3],color='r')
		elif degree == 5:
			plt.plot(plotX,result[3],color='g')
		elif degree ==10:
			plt.plot(plotX,result[3],color='y')
		elif degree ==14:	
			plt.plot(plotX,result[3],color='tomato')
	elif lamba !=0 :
		if lamba ==0.001/20:
			plt.plot(plotX,result[3],color='violet',label="0.0005")
			plt.scatter(plotX,trainY,s=40,c='blue',marker='o',alpha=0.5,label='train')
			plt.scatter(testX,testY,s=40,c='green',marker='x',alpha=0.5,label='test')
		elif lamba ==1/20:
			plt.plot(plotX,result[3],color='g',label="l=0.05")
		elif lamba ==1000/20:
			plt.plot(plotX,result[3],color='navy',label="l=50")
		

