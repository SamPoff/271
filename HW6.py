# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:53:48 2017

@author: Sam
"""

import numpy as np
import math
from math import pi
from scipy.integrate import quad

# f = lambda x: x**2
# quad (f, 0, 1)[0] take first result

"""
Plotter

import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

def f(x):
    return x**2
    
def g(x):
    return x**2 + 1
    
def h(x):
    return np.sin(x)
    
x = np.arange(-10, 5, 0.0001)

plt.plot(x, f(x), 'b')
plt.plot(x, g(x), 'r')
plt.plot(x, g(x), 'y')
"""

"""""""""
###HW6###
"""""""""

"""
#P1
"""


def GaussianElimination(A):

    numrows = len(A)    # Number of rows
    numcols = len(A[0]) # Number of columns

    # Gaussian Elimination
    for col in range(numcols - 1):           
        for row in range(col + 1, numrows):
            # print(A, '\n')
            A[row] = A[row] - (A[row, col]/A[col, col]) * A[col]

    # Make solution array
    x = np.array([0.0,0.0,0.0,0.0])   
        
    # Backsub
    for i in range(numrows - 1, -1, -1): 
        hold = 0.0
        for j in range(i + 1, numrows): 
            hold = hold + A[i,j] * x[j]
        x[i] = (A[i, numcols - 1] - hold) / A[i, i]
        
    return x

"""
x = np.array([0.0,   0.25,    0.5,   0.75,    1.0])
y = np.array([1.0, 1.2840, 1.6487, 2.1170, 2.7183])

def discreetLSP(x, y, degree, xApp):
    
    m = degree + 1
    b       = np.zeros([m,m])
    bCoef   = np.zeros(m)
    equals  = np.zeros(m)
    xExpSum = np.zeros(2 * m)
    xExp    = np.zeros([2 * degree + 1, 2 * degree + 1])
    
    # Find x's to exponents
    for i in range(len(xExp)):
        for j in range(len(xExp)):
            xExp[i,j] = math.pow(x[j],i)
    
    # Find sums of x's to exponents
    for i in range(2 * degree + 1):
        xExpSum[i] = sum(xExp[i,:])
        
    # Find sums of y * x to exponent
    yMult = xExp[0:m,:]
    for i in range(degree + 1):
        for j in range(2 * degree + 1):
            yMult[i,j] = y[j] * yMult[i,j]
    
    # fill b matrix
    for i in range(m):
        for j in range(m):
            b[i,j] = xExpSum[i+j]

    # fill right side
    for i in range(m):
        equals[i] = sum(yMult[i,:])
    
    # concatenate array / matrix
    A = np.array(np.zeros([m,degree+2]))
    for i in range(m):
        for j in range(degree+2):
            if j < m:
                A[i,j] = b[i,j]
            else:
                A[i,j] = equals[i]
    
    # use gaussian elimination to solve
    bCoef = GaussianElimination(A)
    
    # use bCoefs to solve
    yApp = 0
    for i in range(m):
        yApp += bCoef[i] * math.pow(xApp,i)
    
    return yApp

print(discreetLSP(x,y,2,0.8))
"""

"""
#P2
"""

def conLSP(degree):
    
    a = np.zeros(degree + 1)
    xExpInt = np.zeros(2 * degree + 1)
    xSinInt = np.zeros(degree + 1)
    AMat = np.zeros([degree + 1, degree + 2])
    rangeStart = 0
    rangeEnd   = 1
    
    # Fill xExp with exponentiated x's
    for power in range(2 * degree + 1):
        xExpInt[power] = quad(lambda x: x**power, rangeStart, rangeEnd)[0]

    # Fill x times sin integrals
    for power in range(degree + 1):
        xSinInt[power] = quad(lambda x: (x**power)*math.sin(pi*x), rangeStart, rangeEnd)[0]
        
    # Fill AMat with values
    for i in range(degree+1):
        for j in range(degree+2):
            if j != degree + 1:
                AMat[i,j] = xExpInt[i+j]
            else:
                AMat[i,j] = xSinInt[i]
    
    # Solve for coefficients 
    a = GaussianElimination(AMat)
    
    # Set x approximation value and plug into equation
    xApp = 0.5
    approximation = a[0] + a[1]*xApp + a[2]*(xApp**2)
    
    return approximation
    
print(conLSP(2))












