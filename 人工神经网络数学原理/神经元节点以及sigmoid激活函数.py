
import matplotlib.pyplot as plt

# one
x = [1,2,3,4]
y = [0,1,2,3]
plt.step(x,y)
plt.show()

from matplotlib import pylab
import pylab as plt
import numpy as np
# two 
def sigmoid(x):
	return(1 / (1 + np.exp(-x)))
	""" exp(-x)函数，以自然常数e为底的求幂函数，即e^(-x)"""

mysamples = []
mysigmoid =	[]

x = plt.linspace(-10,10,10)
y = plt.linspace(-10,10,100)
plt.plot(x,sigmoid(x),'r',label='linspace(-10,10,10)')
plt.plot(y,sigmoid(y),'r',label='linspace(-10,10,100)')

plt.grid()
plt.title("sigmoid function")
plt.suptitle('sigmoid')

plt.legend(loc='lower right')

plt.text(4,0.8,r'$\sigma(x)=\frac{1}{1+e^(-x)}$', fontsize=15)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

plt.show()