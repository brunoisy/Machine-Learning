import dtree as d
import monkdata as m
import random
import numpy


def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]
n = 200;
nfrac = 6;
fractions = [0.3,0.4,0.5,0.6,0.7,0.8]
E = numpy.zeros((nfrac,n)) #E contains de ratios off correctly classified inputs for each fraction and each iteration
for nf in range(0,nfrac):
	for it in range(0,n):
		monk1train, monk1val = partition(m.monk1, fractions[nf])

		previousPruned = d.buildTree(monk1train,m.attributes)
		previousCheck = d.check(previousPruned, monk1val)
		while(1):
			pruneds = d.allPruned(previousPruned)
			maxCheck = 0
			for pruned in pruneds:
				currentCheck = d.check(pruned, monk1val)
				if(currentCheck>=maxCheck):
					maxCheck = currentCheck
					currentPruned = pruned

			if(maxCheck < previousCheck):
				break
			previousCheck = maxCheck
			previousPruned = currentPruned

		finalPrunned = previousPruned
		E[nf,it] = d.check(finalPrunned, m.monk1test)

means = numpy.mean(E,axis=1)
variances = numpy.var(E,axis=1)
print(means)
print(variances)

for nf in range(0,nfrac):
	for it in range(0,n):
		monk3train, monk3val = partition(m.monk3, fractions[nf])

		previousPruned = d.buildTree(monk3train,m.attributes)
		previousCheck = d.check(previousPruned, monk3val)
		while(1):
			pruneds = d.allPruned(previousPruned)
			maxCheck = 0
			for pruned in pruneds:
				currentCheck = d.check(pruned, monk1val)
				if(currentCheck>=maxCheck):
					maxCheck = currentCheck
					currentPruned = pruned

			if(maxCheck < previousCheck):
				break
			previousCheck = maxCheck
			previousPruned = currentPruned

		finalPrunned = previousPruned
		E[nf,it] = d.check(finalPrunned, m.monk3test)

means = numpy.mean(E,axis=1)
variances = numpy.var(E,axis=1)
print(means)
print(variances)



