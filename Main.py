from Trainer import Trainer

numHidden = 3
numInputsPerHidden = 5
absMaxL = 3

tr = Trainer(numHidden, numInputsPerHidden, absMaxL)
tr.train()
