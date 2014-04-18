# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:29:38 2014

@author: dbell
"""

import argparse
import itertools
import pandas as pd
import os 
import re 
import scipy.io as sio
import numpy as np
import sys

from pandas import Panel, DataFrame, Series
import MP

_NOTEBOOK = False; # change to true if you're going to run this locally 

parser = argparse.ArgumentParser()
parser.add_argument('data', type=unicode)
parser.add_argument('out', type=unicode)
#parser.add_argument('-vi', '--var_info', type=unicode, default='../../data/vpicu/physiologic-variable-normals.csv') # I need to ook at this
parser.add_argument('-rr', '--resample_rate', type=int, default=60)
parser.add_argument('-ma', '--max_age', type=int, default=18)

if _NOTEBOOK:
    args = parser.parse_args( 'settings go here' ) #you're going to have to update with proper params
else:
    args = parser.parse_args()
    
data_dir = args.data
out_dir = args.out
try:
    os.makedirs(out_dir)
except:
    pass

#ranges = DataFrame.from_csv(args.var.info, parse_dates=False)
#ranges = ranges[['Low', 'Normal', 'High']]
#varids = ranges.index.tolist()
resample_rate = '{0}min'.format(args.resample_rate)
max_age = args.max_age

patients = os.listdir(data_dir)
N = len(patients)
Xraw = []
Xmiss = []
X = []

Traw = np.zeros((N,), dtype=int)    # Number of (irregular) samples for episode
T = np.zeros((N,), dtype=int)       # Resampled episode lengths (for convenience)
age = np.zeros((N,), dtype=int)     # Per-episode patient ages in months
gender = np.zeros((N,), dtype=int) # Per-episode patient gender
weight = np.zeros((N,))             # Per-episode patient weight
lmf = np.zeros((N,))                # Last-first duration

idx = 0
for pat in patients:
    try:
        subj = MP.MPSubject.from_file(os.path.join(data_dir, pat))
    except MP.InvalidMPDataException as e:
        if e.field == 'msmts' and e.err == 'size':
            count_nodata += 1
        else:
            sys.stdout.write('\nskipping: ' + str(e))
        continue
    s = subj.as_nparray()
    mylmf  = (s[:,0].max() - s[:,0].min())/60
    
    #store raw time series
    Xraw.append(s)
    lmf[idx] = mylmf
    Traw[idx] = s.shape[0]
    
    #resample
    s = subj.as_nparray_resampled(rate=resample_rate, impute=True)
    if not np.all(~np.isnan(s)):
        #print df
        print pat
    X.append(s)
    
    T[idx] = s.shape[0]
    age[idx] = subj._age
    gender[idx] = subj._gender
    weight[idx] = subj._weight
    
    #y[idx]   = np.nan if ep.died is None else 1 if ep.died else 0
    #pdiag[idx]  = ep.prim_diag
    
    idx += 1
    
#features = features[0:idx,]
#epids = epids[0:idx]
#Traw = Traw[0:idx]
#T = T[0:idx]
#age = age[0:idx]
#gender = gender[0:idx]
#weight = weight[0:idx]
#y = y[0:idx]
#pdiag = pdiag[0:idx]
#los = los[0:idx]
#lmf = lmf[0:idx]
    
    
    
    