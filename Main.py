from Tree import Tree
import numpy as np
import sys

numHidden = 30
N = 50
treeA = Tree(numHidden, N, 5)
treeB = Tree(numHidden, N, 5)
def train(treeA,treeB, N, numHidden):
        same = 0.0
        count = 0
        while(same<0.999):
                inputs = np.random.randint(-5,6, [numHidden, N])
                hiddenA,outputA = treeA.getActivations(inputs)
                hiddenB,outputB = treeB.getActivations(inputs)
                sys.stdout.write('\r' + 'Percent Similar: ' + str(same) +', iteration: ' +str(count))
                sys.stdout.flush()
                same = 0.99*(same) + 0.01*float(outputA == outputB)
                treeA.updateWeights(inputs, hiddenA, outputA, outputB)
                treeB.updateWeights(inputs, hiddenB, outputB, outputA)
                count += 1
        print("\n")



train(treeA, treeB, N, numHidden)
