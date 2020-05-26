import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

xdata = [ 
17,
30,
41,
45,
51,
66,
76,
92,
114,
153,
222,
254,
290,
391,
460,
604,
739,
890,
1020,
1250,
1484,
1706,
2012,
2407,
2637,
2878,
3333,
3645,
4042,
4509,
5077,
5274,
5747,
6171,
6500,
6889,
7491,
8053,
8537,
8990,
9630,
10267,
10998,
11594,
12336,
12970
]

ydata = range(1,47)

def sigmoid(x,L,x0,k,b):
    return( L/(1+np.exp(-k*(x-x0)))+b)

p0 = [max(ydata), np.median(xdata),1,min(ydata)]

popt, pcov = curve_fit( sigmoid, xdata, ydata, p0, method='dogbox')
print( popt )
print( pcov )

x = np.linspace( 1, 50, 50)
y = sigmoid( x, *popt)
plt.plot( xdata,ydata, 'o', label='cases')
plt.plot( x,y, label='fit')
plt.ylim(0, 1300)
plt.legend(loc='best')
