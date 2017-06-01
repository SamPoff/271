import numpy as np
from math import pi

"""
Problem One Jacobi
"""

# Initialize an array of zeros for X
x = np.zeros([80,1]) 
b = []
for i in range(81):
    b.append(pi)
counter = 0

A = np.zeros((80,80))
for i in range(1,81):
    for j in range(1,81):
        if j == i:
            A[i-1,j-1] = 2*i
        elif j == i + 2 and i in range(1,79) or j == i - 2 and i in range(3,81):
            A[i-1,j-1] = 0.5*i
        elif j == i + 4 and i in range(1,77) or j == i - 4 and i in range(5,81):
            A[i-1,j-1] = 0.25*i
        else:
            A[i-1,j-1] = 0.0

# Fill b
for i in range(1,81):
    b[i] = pi

L = 1
LOld = 2

while abs(L - LOld) > 1e-5:
    # Copy old 'x' for x(k-1)
    x_old = x.copy() 
    for i in range(1, 81):
        s = 0.0
        for j in range(1, 81):
            s = s + (A[i-1,j-1] * x_old[j-1])
        
        x[i-1] = (b[i-1] - s) / A[i-1,i-1]
    LOld = max(abs(x_old))
    L = max(abs(x))
    counter += 1

print('\nJacobi\n')
print(counter, ' Iterations\n')
for i in range(10):
    print('x[', i, '] = ', x[i])

    
    
    
    
    
    
    
"""
Problem One GS
"""
# Initialize an array of zeros for X
x = np.zeros([80,1]) 
b = []
for i in range(81):
    b.append(pi)
counter = 0

A = np.zeros((80,80))
for i in range(1,81):
    for j in range(1,81):
        if j == i:
            A[i-1,j-1] = 2*i
        elif j == i + 2 and i in range(1,79) or j == i - 2 and i in range(3,81):
            A[i-1,j-1] = 0.5*i
        elif j == i + 4 and i in range(1,77) or j == i - 4 and i in range(5,81):
            A[i-1,j-1] = 0.25*i
        else:
            A[i-1,j-1] = 0.0

# Fill b
for i in range(1,81):
    b[i] = pi

L = 1
LOld = 2
x_old = x+1

while abs(L - LOld) > 1e-5: 
    # Copy old 'x' for x(k-1)
    x_old = x.copy() 
    for i in range(1, 81):
        s = 0.0
        for j in range(1, 81):
            s = s + (A[i-1,j-1] * x[j-1])
        
        x[i-1] = (b[i-1] - s) / A[i-1,i-1]
    LOld = max(abs(x_old))
    L = max(abs(x))
    #x_old = x.copy()
    counter += 1

print('\nGauss Seidel\n')
print(counter, ' Iterations\n')
for i in range(10):
    print('x[', i, '] = ', x[i])
    
    
    
    
    
    
    
    
    
    
    