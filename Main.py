from Tree import Tree
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser(description='Arguments for neural public key exchange')
parser.add_argument('-numHidden', default = 10 ,type=int, help = 'Number of hidden nodes for the tree parity machine')
parser.add_argument('-inputLength',default = 5 ,type=int, help = 'Number of inputs per hidden node for the tree parity machine')
parser.add_argument('-maxValue', default = 10 ,type=int, help = 'Absolute value for the bounds of the networks weights')
parser.add_argument('-simple', default = True, type = bool, help = 'Indicates whether or not to run a basic key exchange') 
parser.add_argument('-attack',default = False, type = bool, help = 'Indicates whether or not to run a key exchange while being attacked')
parser.add_argument('-tensorboard', default = False, type = bool, help = 'Not supported yet')
args = parser.parse_args()


K = args.numHidden
N = args.inputLength
L = args.maxValue

def log(count, accuracyAB, accuracyE, tensorboard = False):
	#TODO: Add tensorboard support
	if(accuracyE == None):
		sys.stdout.write('\r'+ "Iteration : %d | A/B Accuracy : %f"%(count,accuracyAB))
		#sys.stdout.write('\r' + str(count) +',' +str(accuracyAB))
	sys.stdout.flush()


def getTrees(K,N,L, numTrees):
	return [Tree(K,N,L) for _ in range(numTrees)]


def expMovGen(logging= False):
	accuracy = 0.0
	count = 0 
	while(True):
		updateValue = yield [accuracy, count]
		if(logging):
			log(count, accuracy, None)

		accuracy = accuracy*0.99 + 0.01*updateValue
		count += 1

def regularTrain(K, N, L):
	treeA, treeB  = getTrees(K,N,L,numTrees = 2)
	accuracyManager = expMovGen(logging = True)
	accuracyManager.send(None)
	accuracy = 0.0
	while(accuracy<0.99):
		inputs = np.random.randint(-L,L+1, [K, N])
		hiddenA,outputA = treeA.getActivations(inputs)
		hiddenB,outputB = treeB.getActivations(inputs)
		accuracy, count = accuracyManager.send(float(outputA == outputB))
		treeA.updateWeights(inputs, hiddenA, outputA, outputB)
		treeB.updateWeights(inputs, hiddenB, outputB, outputA)
	print("\nIt took %s iterations for the networks to synchronize."%str(count))
	return [accuracy, count]



def trainWithSimpleAttack(K,N,L):
	treeA, treeB, treeC = getTrees(K,N,L,numTrees=3)
	accuracyManager = expMovGen()
	accuracyManager.send(None)
	accuracy = 0.0
	while(accuracy<0.99):
		inputs = np.random.randint(-L,L+1, [K, N])
		hiddenA,outputA = treeA.getActivations(inputs)
		hiddenB,outputB = treeB.getActivations(inputs)
		accuracy, count = accuracyManager.send(float(outputA == outputB))
		treeA.updateWeights(inputs, hiddenA, outputA, outputB)
		treeB.updateWeights(inputs, hiddenB, outputB, outputA)
	print("\nIt took %s iterations for the networks to synchronize."%str(count))


def experiment(numIters, K,N,L):
	counts = []
	for i in range(numIters):
		avgCount = avgCount + regularTrain(K,N,L)[1]

	avgCounts = np.mean(np.array(counts))
	stdDev = np.std(np.array(counts))

	print("|+++++++EXPERIMENT COMPLETE++++++++|")
	print("|Average Iterations Required : %s  |"%str(avgCounts))
	print("|Standard Deviation          : %s  |"%str(stdDev))
	print("|++++++++++++++++++++++++++++++++++|")

experiment(10, K,N,L)
