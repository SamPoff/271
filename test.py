# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:10:25 2017

@author: Sam
"""

import numpy as np

a = np.array([ [    10,    54.1,   303.4,  1759.8],
               [  54.1,   303.4,  1759.8, 10523.1],
               [ 303.4,  1759.8, 10523.1, 64608.0],
               [1759.8, 10523.1, 64608.0, 40516.7]])

b = np.array([1958.4, 11366.8, 68006.7, 417730.1])

print(a.shape)
#a.reshape((4,4))

print(np.linalg.solve(a,b))