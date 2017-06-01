# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:00:45 2017

@author: Sam
"""

import math
import numpy as np

"""
Problem 4 

Note: dInverse = np.linalg.inv(D)
      xT = x.T


A = np.array([[10., -1,  2,  0],
              [ -1, 11, -1,  3],
              [  2, -1, 10, -1],
              [  0,  3, -1,  8]])
     
b = np.array([[  6. ],
              [ 25 ],
              [-11 ],
              [ 15 ]])
     
rows = 4
cols = 4

D     = np.zeros([rows,cols])
L     = np.zeros([rows,cols])
U     = np.zeros([rows,cols])
X     = np.zeros([rows,   1])
X_Old = np.zeros([rows,   1])

# Set D
for i in range(rows):
    for j in range(cols):
        if i == j:
            D[i,j] = A[i,j]

# Set L
for i in range(rows):
    for j in range(cols):
        if i > 0 and j < i:
            L[i,j] = -A[i,j] 
  
# Set U
for i in range(rows):
    for j in range(cols):
        if j > 0 and i < j:
            U[i,j] = -A[i,j]

# Set D inverse
DI = np.linalg.inv(D)

# Set tolerance
tol = 1e-10

# Iterate for X
for i in range(1000):
    X = np.dot(DI, np.dot((L + U), X_Old)) + np.dot(DI, b)
    # Check tolerance norms
    delta = np.linalg.norm((X - X_Old),np.inf) / np.linalg.norm(X,np.inf)
    # Set old X
    X_Old = X

    # Break when tolerance is met
    if(delta < tol):
        # Display number of iterations
        print('\nIterations: ', i+1, '\n')
        break
    
# Print X and A.X
print('X = \n', X, '\n')
print('A.X = \n', np.dot(A,X), '\n')
    
"""

"""
Problem 5
"""
"""
A = np.array([[1, 1, 1],
              [0, 1, 1],
              [0, 0, 1]])

A = np.array([[1, 0, 1],
              [1, 2, 0],
              [1, 2, 0],
              [1, 2, 1]])

def GramSchmidt(A):
    
    rows = len(A)
    cols = len(A[0])
    
    Q  = np.zeros([rows, cols])
    Q[:,0] = A[:,0]
    
    for k in range(1,cols):
        
        s = np.zeros(rows)

        for i in range(k):
            
            s = s + (np.dot(Q[:,i], A[:,k])) / (np.dot(Q[:,i], Q[:,i])) * Q[:,i]
            
            
        Q[:,k] = A[:,k] - s

    return Q
    
print('Q:\n',GramSchmidt(A))  
"""

"""
Problem 6
"""

A = np.array([[1, 0, 1],
              [1, 2, 0],
              [1, 2, 0],
              [1, 2, 1]])

def GramSchmidtOrth(A):
    
    rows = len(A)
    cols = len(A[0])
    
    Q  = np.zeros([rows, cols])
    Q[:,0] = A[:,0]
    
    for k in range(1,cols):
        
        s = np.zeros(rows)

        for i in range(k):
            
            s = s + (np.dot(Q[:,i], A[:,k])) / (np.dot(Q[:,i], Q[:,i])) * Q[:,i]
            
            
        Q[:,k] = A[:,k] - s

    print('Q:\n',Q)

    for i in range(cols):
        
        norm = np.linalg.norm(Q[:,i],2)
        
        for j in range(rows):
            
            Q[j,i] = Q[j,i] / norm

    return Q
    
print('\nOrthogonal:\n',GramSchmidtOrth(A)) 

    
    
    