# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:56:35 2017

@author: Sam
"""
import numpy as np
import math

"""
P1
"""

"""
x = np.array([-1,  0, 1,  2])
y = np.array([ 3, -4, 5, -6])
"""

"""
x = np.array([  2,   2.75,   4])
y = np.array([1/2, 1/2.75, 1/4])


def LIP(x, y, xApp):
    
    # Get n and fill l with ones.
    n = len(x)
    l = np.ones(n)
    
    # i is the l number we are on, j is the x
    for i in range(n):
        for j in range(n):
            if i != j:
                l[i] *= (xApp - x[j]) / (x[i] - x[j])
         
    # Set yApp to zero and sum l and y parts.
    yApp = 0
    for i in range(n):
        yApp += l[i] * y[i]
    
    return yApp
    
print(LIP(x, y, 3))
"""


"""
P2
"""


"""
def devidedDif(x, y, xApp):
    
    n = len(x)
    yApp = 0
    a = np.zeros([n,n])
    a[:,0] = y
    
    for j in range(1,n):
        for i in range(j,n):
            a[i,j] = (a[i,j-1] - a[i-1,j-1]) / (x[i] - x[i-j])
        
    for i in range(n):
        hold = 1.0
        for j in range(i):
            hold *= xApp - x[j]

        yApp += a[i,i] * hold
        
    return yApp

print(devidedDif(x,y,1.5))
"""


"""
P3
"""

"""
def leastSquares(x, y, xApp):
    
    xSq = np.zeros(len(x))
    for i in range(len(x)):
        xSq[i] = math.pow(x[i],2)
    
    b0Top    = ( sum(xSq) * sum(y) ) - ( sum(x*y) * sum(x) )
    bottom = ( len(x) * sum(xSq) - math.pow(sum(x),2) )
    b0       =   b0Top / bottom
    
    b1Top    = ( len(x) * sum(x*y) - (sum(x) * sum(y)) )
    b1       =   b1Top / bottom
    
    yApp = b0 + b1 * xApp
    
    return yApp
    
x = np.array([  1,   2,   3, 4, 5,   6,    7,    8,  9,   10]) 
y = np.array([1.3, 3.5, 4.2, 5, 7, 8.8, 10.1, 12.5, 13, 15.6])    
xApp = 1.5
print(leastSquares(x,y,xApp))
"""


"""
Number Recognition
"""

"""
def E(x, y, a, b):
    

def grad_alpha(x, y, a, b):
    
    
def grad_beta(x, y, a, b):
    
    
def linear_regression(x, y):
    
    p = len(x)
    aOld = 0
    bOld = np.zeros(p)
    gamma = 0.001
    max_iterations = 2000
    
    for i in range(max_iterations):
        aNew = aOld - (gamma * grad_alpha(x, y, aOld, bOld))
        for j in range(p):
            bOld = bOld - (gamma * grad_beta(x, y, aOld, bOld))
        Enew = E(x, y, aNew, bNew)
        if abs(Enew - Eold) < 1e-5:
"""         
    
a = np.matrix([[    10,    54.1,   303.4,  1759.8],
               [  54.1,   303.4,  1759.8, 10523.1],
               [ 303.4,  1759,8, 10523.1, 64608.0],
               [1759.8, 10523.1, 64608.0, 40516.7] ])

b = np.array([1958.4, 11366.8, 68006.7, 417730.1])

print(a.shape)
print(b.shape)
print(a)
#print(np.linalg.solve(a,b))
    
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
    
print(GaussianElimination(a))
"""
    
    
    
    
    




