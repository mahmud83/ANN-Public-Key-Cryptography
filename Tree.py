import numpy as np
import sys

class Tree():
	def __init__(self,k,n,l):
		self.generateTree(k,n,l)	
		self.l = l

	def generateTree(self, k,n,l):
		self.weights = np.random.randint(-l,l+1, [k, n])


	def getActivations(self, inputs, nonActivated = False):
		hidden = []
		for weightGroup, inputGroup in zip(self.weights, inputs):
			hidden.append(np.dot(weightGroup, inputGroup))

		if(nonActivated):
			return np.abs(np.array(hidden))
            
		hidden = np.sign(np.array(hidden))
		output = np.prod(hidden)
		return [hidden, output]
		

	def updateWeights(self,inputs, hidden, outputSelf, outputOther):
		for index, input in enumerate(inputs):
			update = input*float(hidden[index]==outputSelf)*float(outputSelf==outputOther==1)
			self.weights = np.clip((self.weights+update), -self.l, self.l)

    

