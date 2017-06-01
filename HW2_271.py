# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:08:24 2017

@author: Sam
"""


import numpy as np
import math 

A = np.array([[math.pi             , -math.exp(1) , math.sqrt(2)  , -math.sqrt(3) , math.sqrt(11)], 
              [math.pow(math.pi, 2), math.exp(1)  , -math.exp(2)  , (3.0/7.0)     , 0            ],
              [math.sqrt(5)        , -math.sqrt(6), 1             , -math.sqrt(2) , math.pi      ], 
              [math.pow(math.pi, 3), math.exp(2)  , -math.sqrt(7) , (1.0/9.0)     , math.sqrt(2)]])
"""
A = np.array([[6,5,6,5,7], 
              [5,6,7,0,2],
              [3,0,0,1,6], 
              [6,7,8,2,1]])
"""
print('Original\n', A, '\n')

numrows = 4
numcols = 5

for col in range(numcols - 1):
    
    # Pivoting
    maxRow = np.argmax(abs(A[col:numrows, col]))
    temp = A[maxRow + col].copy()
    A[maxRow + col] = A[col]
    A[col] = temp
    
    for row in range(col + 1, numrows):
        print(A, '\n')
        A[row] = A[row] - (A[row, col]/A[col, col]) * A[col]
        
print('\n', A)
 
x = np.array([0.0,0.0,0.0,0.0])
        
for i in range(numrows - 1, -1, -1):
    hold = 0.0
    for j in range(i + 1, numrows): 
        hold = hold + A[i,j] * x[j]
    x[i] = (A[i, numcols - 1] - hold) / A[i, i]
    
for z in range(4):
    print('\nX[',z,'] =', x[z])

# pivoting
# np.argmax(abs(A[:,0])) # should output the position (index) of the max value.
# determinate
# np.linalg.det(A) # gives the determinate of A