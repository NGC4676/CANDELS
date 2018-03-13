# -*- coding: utf-8 -*-
"""
Created on Fri Sep 01 01:28:27 2017

@author: Qing Liu
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# (Papovich 2015) Abundance Matching MW Progenitor
UV_pop = np.array([0.6,0.6,0.9,1.1,1.4,1.6,1.7,1.9])
VJ_pop = np.array([0.3,0.5,0.8,1.0,1.2,1.2,1.3,1.3])
Ms_pop = np.array([9.48,9.7,9.88,10.06,10.21,10.35,10.47,10.6])
z_pop = np.array([2.5,2.1,1.85,1.55,1.25,1.0,0.8,0.45])

# UVJ
plt.figure(figsize=(6,6))
plt.plot(VJ_pop, UV_pop)
s = plt.scatter(VJ_pop, UV_pop)
plt.plot([-1., 1.0, 1.6, 1.6], [1.3, 1.3, 2.01, 2.5], color='k', alpha=0.7)
plt.xlabel('V - J',fontsize=12)
plt.ylabel('U - V',fontsize=12)
plt.xlim([0.0, 2.0])
plt.ylim([0.0, 2.25])
plt.show()


# M* - z
plt.figure(figsize=(6,6))
plt.plot(Ms_pop, z_pop,'k')
s = plt.scatter(Ms_pop, z_pop, c='k')
plt.xlabel('M*',fontsize=12)
plt.ylabel('z',fontsize=12)
plt.xlim([9.0, 11.0])
plt.ylim([0.0, 2.6])
plt.gca().invert_yaxis()
plt.show()

print np.interp(2.25, z_pop[::-1], Ms_pop[::-1])
print np.interp(0.75, z_pop[::-1], Ms_pop[::-1])
