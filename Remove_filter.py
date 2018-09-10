#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 17:56:52 2018

@author: Q.Liu
"""

import numpy as np
import pandas as pd

from astropy.io import ascii

fname = "N/Aldo/Aldo_SFH"
table = np.loadtxt(fname+".cat")

table_new = np.hstack([table[:,0][:,None],table[:,5:]])

colnames=["id", "F1", "E1","F2", "E2", "F3", "E3", "F4", "E4",
          "F5", "E5", "F6", "E6", "F7", "E7", "F8", "E8", "zspec"]

np.savetxt(fname+"_nouv.cat", table_new, header=" ".join(colnames),
           fmt=['%d']+['%.5e' for i in range(16)]+['%.2f'])