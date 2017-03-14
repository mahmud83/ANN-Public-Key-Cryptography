import numpy as np
from Tree import Tree
from random import randint

class Trainer():
	def __init__(self, numHidden, numInputs, L):
		self.trees = {name:Tree(numHidden, numInputs, L) for name in ['A','B','E']}
		self.numHidden = numHidden
		self.numInputs = numInputs

	def train(self,printEveryN = 300 ,testing = False):
		expAvgErrors = 1.
		iters = 0
		while(expAvgErrors > 0.05):
			input = np.array([[randint(0,1) for _ in range(self.numInputs)] for _ in range(self.numHidden)])
			input[input == 0] = -1
			[pA,hA], [pB,hB], [pC,hC] = [tree.predict(input) for tree in self.trees.values()]
			expAvgErrors = expAvgErrors*0.999 + 0.001* (pA!=pB)
			if(pA==pB):
				for index, logits in enumerate(zip(hA, hB)):
					if(logits[0] == pA):
						self.trees['A'].updateWeights(input[index], index)
					if(logits[1] == pB):
						self.trees['B'].updateWeights(input[index], index)
						
			iters += 1
			if(iters %printEveryN == 0):
				print("Iteration %d, Error %f"%(iters, expAvgErrors))		
			
			

