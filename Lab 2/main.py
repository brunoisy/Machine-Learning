from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy , pylab , random , math

# Generating Test Data
numpy.random.seed(100)
classA = [(random.normalvariate(-1.5, 1),
           random.normalvariate(0.5, 1),
           1.0)
          for i in range(5)] + \
         [(random.normalvariate(1.5, 1),
           random.normalvariate(0.5, 1),
           1.0)
          for i in range(5)]

classB = [(random.normalvariate(0.0, 0.5),
           random.normalvariate(-0.5, 0.5),
           -1.0)
          for i in range(10)]

data = classA + classB
random.shuffle(data)


####################################################################
# Implementation
### Kernel Functions
def linKernel(x, y):
	return x[0]*y[0]+x[1]*y[1]+1

kernel = linKernel


### Matrix P
def buildP(data):
	N = len(data)
	P = numpy.zeros((N,N))
	for i in range(N):
		xi = data[i]
		for j in range(N):			
			xj = data[j]
			P[i,j] = xi[2]*xj[2]*kernel(xi[0:2], xj[0:2])
	return P
P = buildP(data)
print(P)

### Vectors q and h, matrix G
N = len(data)
q = numpy.zeros((N,1))
for i in range(N):
	q[i] = -1
h = numpy.zeros((N,1))
G = numpy.zeros((N,N))
for i in range(N):
	G[i,i]=-1


### Call qp
r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])


### Non zero alphas
eps = 10^-5
nonZeroAlpha = []
for i in range(N):
	if alpha[i]>eps:
		nonZeroAlpha.append(alpha[i])
print(alpha)
print(nonZeroAlpha)


### Indicator function

#####################################################################
## Plotting Test Data
#pylab.hold(True)
#pylab.plot([p[0] for p in classA],
#           [p[1] for p in classA],
#           'bo')
#pylab.plot([p[0] for p in classB],
#           [p[1] for p in classB],
#           'ro')
#pylab.show()



## Plotting decision boundary
#xrange=numpy.arange(-4, 4, 0.05)
#yrange=numpy.arange(-4, 4, 0.05)

#grid=matrix([[indicator(x, y)
#              for y in yrange]
#             for x in xrange])

#pylab.contour(xrange, yrange, grid,
#              (-1.0, 0.0, 1.0),
#              colors=('red', 'black', 'blue'),
#              linewidths=(1, 3, 1))
