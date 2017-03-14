import numpy as np
from random import randint

#ScaledInputBits just means bits where the 0 bit = -1
#List is a nestedList of inputs where each sub-list is the size of numInputs. ie [[-1,1,-1,1,1,1,-1],[-1,1,1,1,-1,-1,1,-1,-1]] 

class Tree():
	def __init__(self, numHidden, numInputs, L):
		self.numNodes = 0
		self.weights = []
		self.lcs = [self.hiddenUnit(L,numInputs) for _ in range(numHidden)]	
		self.output = self.createModel(self.lcs)		

	def hiddenUnit(self, L, numInputs):
		self.weights.append(np.array([randint(-L,L) for _ in range(numInputs)]))
		linearCombination =  lambda scaledInputBits, index: np.sum( scaledInputBits + self.weights[index])
		self.numNodes += 1
		return linearCombination

	def createModel(self, linearCombinations):
		self.weights = np.array(self.weights)
		hiddenUnits = lambda scaledInputBitsList: [np.sign(lc(scaledInputBitsList[index], index)) 
							for index,lc in enumerate(linearCombinations)]
		outputBit = lambda scaledInputBitsList: [np.multiply.reduce(hiddenUnits(scaledInputBitsList)),
								hiddenUnits(scaledInputBitsList)]
		return outputBit

	def predict(self, inputVectors, train = True):
		pred, hidden =  self.output(inputVectors)
		if(train):
			return [pred, hidden]
		else:
			return pred

	def updateWeights(self, vector, index):
		self.weights[index] += vector



