from matplotlib.pylab import  plot, arange, cos, show
import math


X = arange(-10,10,0.01)
Y = abs(X-1) + abs(X+1)
plot(X,Y)
show()