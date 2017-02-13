import dtree
import monkdata as m
import numpy
import random


# Ass. 1
print("Entropy of monk1 :",dtree.entropy(m.monk1))
print("Entropy of monk2 :",dtree.entropy(m.monk2))
print("Entropy of monk3 :",dtree.entropy(m.monk3))


# Ass. 3
infoGains = numpy.zeros((3,6))
monks = [m.monk1, m.monk2, m.monk3]
for i in range(0,3):
	for j in range(0, 6):
		infoGains[i,j] = dtree.averageGain(monks[i], m.attributes[j])
	
print("Information gains :\n", infoGains)


# Ass. 5
infoGains2 = numpy.zeros((4, 6))
subsets = [0 for x in range(0,4)]
for i in range(1,5):
	subsets[i-1] = dtree.select(m.monk1, m.attributes[4], i) 
	for j in range(0, 6):
		infoGains2[i-1,j] = dtree.averageGain(subsets[i-1], m.attributes[j])


print("Information gains for each subset:\n", infoGains2)

# subset[1]
subsets1 = [0 for x in range(0,3)]#a5==2
for i in range(1,4):
	subsets1[i-1] = dtree.select(m.monk1, m.attributes[3], i) 
	print(dtree.mostCommon(subsets1[i-1]))


subsets2 = [0 for x in range(0,2)]
for i in range(1,3):
	subsets2[i-1] = dtree.select(m.monk1, m.attributes[5], i) 
	print(dtree.mostCommon(subsets2[i-1]))


subsets3 = [0 for x in range(0,3)]
for i in range(1,4):
	subsets3[i-1] = dtree.select(m.monk1, m.attributes[0], i) 
	print(dtree.mostCommon(subsets3[i-1]))



print(dtree.buildTree(m.monk1, m.attributes))# problem! inconsistent
#drawtree_qt5.drawTree(dtree.buildTree(m.monk1, m.attributes, 2))




t=dtree.buildTree(m.monk1, m.attributes);
print(dtree.check(t, m.monk1))
t=dtree.buildTree(m.monk2, m.attributes);
print(dtree.check(t, m.monk2))
t=dtree.buildTree(m.monk3, m.attributes);
print(dtree.check(t, m.monk3))

t=dtree.buildTree(m.monk1, m.attributes);
print(dtree.check(t, m.monk1test))
t=dtree.buildTree(m.monk2, m.attributes);
print(dtree.check(t, m.monk2test))
t=dtree.buildTree(m.monk3, m.attributes);
print(dtree.check(t, m.monk3test))

###
#Ass. 6

def partition(data, fraction):
	ldata = list(data)
	random.shuffle(ldata)
	breakPoint = int(len(ldata) * fraction)
	return ldata[:breakPoint], ldata[breakPoint:]

monk1train, monk1val = partition(m.monk1, 0.6)

previousPrunedTree = dtree.buildTree(monk1train,m.attributes)
previousCheck = dtree.check(previousPrunedTree, monk1val)

while(1):
	prunedTrees = dtree.allPruned(previousPrunedTree)
	maxCheck = 0
	for prunedTree in prunedTrees:
		currentCheck = dtree.check(prunedTree, monk1val)
		if(currentCheck>=maxCheck):
			maxCheck = currentCheck
			currentPrunedTree = prunedTree

	if(maxCheck < previousCheck):
		break
	previousCheck = maxCheck
	previousPrunedTree = currentPrunedTree

finalPrunnedTree = previousPrunedTree
print(finalPrunnedTree)
print(dtree.check(finalPrunnedTree, monk1val))







