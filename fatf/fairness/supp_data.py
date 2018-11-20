# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:48:29 2018

@author: rp13102
"""
import numpy as np

def getrow():
    return (feature1_unique[np.random.randint(2, size=1)[0]], 
            feature2_unique[np.random.randint(2, size=1)[0]], 
            protected[np.random.randint(2, size=1)[0]],
            target[np.random.randint(2, size=1)[0]],
            predictions[np.random.randint(2, size=1)[0]]
            )
    
def generatetable(nrows = 10):        
    dt = [('feature1', '<i4'), # 20 character string 
          ('feature2', '<i4'),
          ('gender', '<i4'), # np.int32
          ('target', '<i4'),
          ('prediction', '<i4')
          ]
    
    D = np.array(getrow(), dtype= dt)
    for i in range(nrows):
        c = np.array(getrow(), dtype=dt)
        D = np.append(D, c)   
        
    return D    

feature1_unique = [0, 1]
feature2_unique = [0, 1]
protected = [0, 1]
target = [0, 1]
predictions = [0, 1]


