from Tree import Tree
import numpy as np
import sys
import argparse
from multiprocessing import Process, Queue
import time

parser = argparse.ArgumentParser(description='Arguments for neural public key exchange')
parser.add_argument('-numHidden', default = 10 ,type=int, help = 'Number of hidden nodes for the tree parity machine')
parser.add_argument('-inputLength',default = 5 ,type=int, help = 'Number of inputs per hidden node for the tree parity machine')
parser.add_argument('-maxValue', default = 10 ,type=int, help = 'Absolute value for the bounds of the networks weights')
parser.add_argument('-simple', default = True, type = bool, help = 'Indicates whether or not to run a basic key exchange') 
parser.add_argument('-attack',default = False, type = bool, help = 'Indicates whether or not to run a key exchange while being attacked')
parser.add_argument('-tensorboard', action = 'store_false',  help = 'Not supported yet')
parser.add_argument('-concurrent', action = 'store_true',  help = "Run the experiment concurrently or not")
args = parser.parse_args()

K = args.numHidden
N = args.inputLength
L = args.maxValue

def log(count, accuracyAB, accuracyE, tensorboard = False):
	sys.stdout.write('\r'+ "Iteration : %d | A/B Accuracy : %f | C Accuracy : %f"%(count,accuracyAB, accuracyE))
	sys.stdout.flush()

def getTrees(K,N,L, numTrees):
	return [Tree(K,N,L) for _ in range(numTrees)]

def expMovGen(logging= True):
	accuracyAB, accuracyC = 0.0, 0.0
	count = 0 
	while(True):
		updateValueAB, updateValueC = yield [accuracyAB, accuracyC, count]
		if(logging):
			log(count, accuracyAB, accuracyC)

		accuracyAB = accuracyAB*0.99 + 0.01*updateValueAB
		accuracyC = accuracyC*0.99 + 0.01*updateValueC
		count += 1

def regularTrain(K, N, L, queue,id):
	np.random.seed()
	treeA, treeB  = getTrees(K,N,L,numTrees = 2)
	accuracyManager = expMovGen(logging = True)
	accuracyManager.send(None)
	accuracy = 0.0
	while(accuracy<0.99):
		inputs = np.random.randint(-L,L+1, [K, N])
		hiddenA,outputA = treeA.getActivations(inputs)
		hiddenB,outputB = treeB.getActivations(inputs)
		accuracy,_ ,count = accuracyManager.send([float(outputA == outputB), -1.0])
		treeA.updateWeights(inputs, hiddenA, outputA, outputB)
		treeB.updateWeights(inputs, hiddenB, outputB, outputA)
	print("\nIt took %s iterations for the networks to synchronize."%str(count))
	if(queue != None):
		queue.put(count)

	return [accuracy, count]

def trainWithSimpleAttack(K,N,L,queue, id):
	np.random.seed()
	treeA, treeB, treeC = getTrees(K,N,L,numTrees=3)
	accuracyManager = expMovGen()
	accuracyManager.send(None)
	accuracyAB = 0.0
	while(accuracyAB<0.99):
		inputs = np.random.randint(-L,L+1, [K, N])
		hiddenA,outputA = treeA.getActivations(inputs)
		hiddenB,outputB = treeB.getActivations(inputs)
		hiddenC,outputC = treeC.getActivations(inputs)
		accuracyAB,accuracyC, count = accuracyManager.send([float(outputA == outputB), float(outputA == outputB == outputC)])
		treeA.updateWeights(inputs, hiddenA, outputA, outputB)
		treeB.updateWeights(inputs, hiddenB, outputB, outputA)
		treeC.updateWeights(inputs, hiddenC, outputA, int(outputB == outputC))
	print("\nIt took %s iterations for the networks to synchronize."%str(count))
	if(queue != None):
		queue.put([count, accuracyC])
	return [accuracyAB, accuracyC, count ]

def experiment(numIters, K,N,L, concurrent):
	countQ = Queue()
	counts = []
	eveSync = []
	start = time.time()
	trainFunction = trainWithSimpleAttack
	if(concurrent):
		concurrentGames = [Process(target = trainFunction, args = (K,N,L,countQ, i)) for i in range(numIters)]
		for game in concurrentGames:
			game.start()

		for game in concurrentGames:
			game.join()
	else:
		for i in range(numIters):
			counts.append(trainFunction(K,N,L, None, i)[1])

	while(not countQ.empty()):
		c,e = countQ.get()
		counts.append(c)
		eveSync.append(e)

	elapsed = time.time() - start
	
	print(counts)
	avgCounts = np.mean(np.array(counts))
	stdDev = np.std(np.array(counts))
	avgEve = np.mean(np.array(eveSync))
	stdDevEve = np.std(np.array(eveSync))

	print("|+++++++EXPERIMENT COMPLETE++++++++|")
	print("|Average Iterations Required : %s  |"%str(avgCounts))
	print("|Standard Deviation AB Sync  : %s  |"%str(stdDev))
	print("|Average Eve Synchronization : %s  |"%str(avgEve))
	print("|Standard Deviation Eve Sync : %s  |"%str(stdDevEve))
	print("|Elapsed Time                : %s  |"%str(elapsed))
	print("|++++++++++++++++++++++++++++++++++|")

experiment(4, K,N,L, args.concurrent)
