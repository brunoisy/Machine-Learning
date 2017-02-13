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

p = 4
def polyKernel(x, y):
	return (x[0]*y[0]+x[1]*y[1]+1)**p

sigma = 1
def rbfKernel(x,y):
	return numpy.exp(-((x[0]-y[0])**2+(x[1]-y[1])**2)/(2*sigma**2))

k = 1
delta = 1
def sigmoidKernel(x,y):
	return numpy.tanh(k*(x[0]*y[0]+x[1]*y[1])-delta)

kernel = polyKernel


### Matrix P
N = len(data)
P = numpy.zeros((N,N))
for i in range(N):
	xi = data[i]
	for j in range(N):			
		xj = data[j]
		P[i,j] = xi[2]*xj[2]*kernel(xi[0:2], xj[0:2])


### Vectors q and h, matrix G
C = 1
q = numpy.zeros((N,1))
for i in range(N):
	q[i] = -1
h = numpy.zeros((2*N,1))
for i in range(N):
	h[N+i] = C
G = numpy.zeros((2*N,N))
for i in range(N):
	G[i,i]=-1
for i in range(N):
	G[N+i,i]=1


### Call qp
r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
alpha = list(r['x'])


### Non zero alphas
eps = 10**(-5)
print(eps)
nonZeroAlphas = []
datas = []
for i in range(N):
	if alpha[i]>eps:
		nonZeroAlphas.append(alpha[i])
		datas.append(data[i])
print(alpha)
print(nonZeroAlphas)


### Indicator function
def indicator(x, y):
	indx = 0
	for i in range(len(nonZeroAlphas)):
		indx += nonZeroAlphas[i]*datas[i][2]*kernel([x, y], datas[i][0:2])
	return indx


#####################################################################
# Plotting Test Data
pylab.hold(True)
pylab.plot([p[0] for p in classA],
           [p[1] for p in classA],
           'bo')
pylab.plot([p[0] for p in classB],
           [p[1] for p in classB],
           'ro')




# Plotting decision boundary
xrange=numpy.arange(-4, 4, 0.05)
yrange=numpy.arange(-4, 4, 0.05)


grid=matrix([[indicator(x, y)
              for y in yrange]
             for x in xrange])

pylab.contour(xrange, yrange, grid,
              (-1.0, 0.0, 1.0),
              colors=('red', 'black', 'blue'),
              linewidths=(1, 3, 1))

pylab.show()
