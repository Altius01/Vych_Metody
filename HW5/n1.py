#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 23:16:12 2020

@author: altius01
"""

import numpy as np
import matplotlib.pyplot as plt

with np.load('data.npz') as data:
    A, C = data['A'], data['C']
    
plt.imshow(A)