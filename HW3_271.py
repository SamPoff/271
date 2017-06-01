import numpy as np

"""
Solving a matrix using LU factorization.
"""

print('LU\n')

# Set rows and columns
row = 4
col = 5
A = np.random.rand(row, col)
print('\nOriginal A:\n', A)

# Create 'b' array
b = np.zeros([row,1])
for i in range(row):
    b[i,0] = A[i,col-1]

def LU(A, b):

    rowsA = len(A)    # Rows in array
    colsA = len(A[0]) # Cols in array

    # Initialize new array with all zeros
    upper = np.zeros((rowsA, colsA - 1))

    # Add in values from old array
    for i in range(rowsA):
        for j in range(colsA - 1):
            upper[i,j] = A[i,j]
    
    # Make array for lower
    lower = np.eye(rowsA)
    
    # Use gaussian elimination to get upper triangular
    for col in range(colsA - 2):           
        for row in range(col + 1, rowsA):
            
            # Gaussian Elimination saving the m's
            m = (upper[row, col]/upper[col, col])
            upper[row] = upper[row] - m * upper[col]
    
            # Add m's to lower matrix
            lower[row,col] = m
    
    # Check that lower.upper == original
    #check = np.dot(lower,upper)
    #print('\nCheck: \n',check)
    
    # Make array to hold y's
    y = np.zeros([rowsA,1])
    
    # Forward sub
    for i in range(rowsA):
        s = 0
        for j in range(0,i):
            s = s + lower[i,j] * y[j]
        y[i] = b[i] - s
    
    print('\nLower dot Y\n', np.dot(lower,y))
    
    # Initialize an array of zeros for X
    x = np.zeros([rowsA,1])
        
    # Back sub
    for i in range(rowsA - 1, -1, -1):
        hold = 0.0
        for j in range(i + 1, rowsA): 
            hold = hold + upper[i,j] * x[j]
        x[i] = (y[i] - hold) / upper[i, i]

    print('\nX = \n', x)
    print('\nAX = \n', np.dot(A[:,0:4],x))
    
    return x

LU(A,b)








"""
Solving a matrix using Gauss Seidel method.
"""

print('\nGauss-Seidel')

row = 4 
col = 5 

A = np.array([[10, -1,  2,  0,   6], 
              [-1, 11, -1,  3,  25], 
              [2 , -1, 10, -1, -11], 
              [0 ,  3, -1,  8,  15]])

print('\nOriginal A:\n', A)

# Create 'b' array
b = np.zeros([row,1])
for i in range(row):
    b[i,0] = A[i,col-1]

def GaussS(A,b):
    
    A = A * 1.0
    rowsA = len(A)    # Rows in array
    colsA = len(A[0]) # Cols in array
    
    # Initialize an array of zeros for X, s for sum
    x = np.zeros([rowsA,1])
    s = 0.0
    
    for k in range(1,100):
        for i in range(rowsA):
            s = 0.0
            for j in range(colsA - 1):
                if j != i:
                    s = s + (A[i,j] * x[j])
                else:
                    continue

            x[i] = (b[i] - s) / A[i,i]
    
    return x

x = GaussS(A,b)
print('\nX = \n', x)
print('\nAX = \n', np.dot(A[:,0:4],x))









"""
Solving a matrix using Jacobi method.
"""

print('\nJacobi')

row = 4 
col = 5 

A = np.array([[10, -1,  2,  0,   6], 
              [-1, 11, -1,  3,  25], 
              [2 , -1, 10, -1, -11], 
              [0 ,  3, -1,  8,  15]])

print('\nOriginal A:\n', A)

# Create 'b' array
b = np.zeros([row,1])
for i in range(row):
    b[i,0] = A[i,col-1]

def jacobi(A,b):
    
    A = A * 1.0
    rowsA = len(A)    # Rows in array
    colsA = len(A[0]) # Cols in array
    
    # Initialize an array of zeros for X
    x = np.zeros([rowsA,1]) 
    
    for k in range(1,100):
        # Copy old 'x' for x(k-1)
        x_old = x.copy() 
        for i in range(rowsA):
            s = 0.0
            for j in range(colsA - 1):
                if j != i:
                    s = s + (A[i,j] * x_old[j])
                else:
                    continue

            x[i] = (b[i] - s) / A[i,i]
    
    return x

x = jacobi(A,b)
print('\nX = \n', x)
print('\nAX = \n', np.dot(A[:,0:4],x))


