# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:29:38 2014

@author: dbell
@author: davekale
"""

from __future__ import division

import argparse
import glob
import os
import sys

import numpy as np
import scipy.io as sio

from pydatasets.physionet import challenge2012
from pydatasets.physionet.challenge2012 import Challenge2012Episode


parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=unicode)
parser.add_argument('out_dir', type=unicode)
parser.add_argument('-b', '--basename', type=unicode, default='physionet_challenge2012')
parser.add_argument('-v', '--variables', type=unicode, nargs='+', default=['ALP', 'ALT', 'AST', 'Albumin', 'BUN',
                                                                           'Bilirubin', 'Cholesterol', 'Creatinine',
                                                                           'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3',
                                                                           'HCT', 'HR', 'K', 'Lactate', 'MAP',
                                                                           'MechVent', 'Mg', 'NIDiasABP', 'NIMAP',
                                                                           'NISysABP', 'Na', 'PaCO2', 'PaO2',
                                                                           'Platelets', 'RespRate', 'SaO2', 'SysABP',
                                                                           'Temp', 'TroponinI', 'TroponinT', 'Urine',
                                                                           'WBC', 'Weight', 'pH'])
parser.add_argument('-r', '--resample_rate', type=int, default=60)
parser.add_argument('--merge_bp', action='store_true')
args = parser.parse_args()
args.variables = set(args.variables)

fns = glob.glob(os.path.join(os.path.join(args.data_dir, 'set-a'), '*.txt'))
fns.extend(glob.glob(os.path.join(os.path.join(args.data_dir, 'set-b'), '*.txt')))
#eps = [ Challenge2012Episode.from_file(fn, args.variables) for fn in fns ]
eps = []
sentinel = 0.0
for i, fn in enumerate(fns):
    if i / len(fns) > sentinel:
        sys.stdout.write('.')
        sys.stdout.flush()
        sentinel += 0.01
    eps.append(Challenge2012Episode.from_file(fn, args.variables))
sys.stdout.write('\n')
eps = challenge2012.add_outcomes(eps, os.path.join(args.data_dir, 'Outcomes-a.txt'))

variables = args.variables
if args.merge_bp:
    to_merge = { 'SysABP': ('NISysABP', 'SysABP'), 'DiasABP': ('NIDiasABP', 'DiasABP'), 'MAP': ('NIMAP', 'MAP') }
    variables = None
    for ep in eps:
        ep.merge_variables(to_merge)
        variables = set(ep._data.columns.tolist()) if variables is None else variables
variables = sorted(variables)

Xraw  = []
Traw  = np.zeros((len(eps),), dtype=int)
tsraw = []
Xmiss = []
X     = []
T     = np.zeros((len(eps),), dtype=int)

recordid = np.zeros((len(eps),), dtype=int)
age      = np.zeros((len(eps),), dtype=int)
gender   = np.zeros((len(eps),), dtype=int)
height   = np.zeros((len(eps),))
weight   = np.zeros((len(eps),))
icutype  = np.zeros((len(eps),), dtype=int)
source   = np.zeros((len(eps),), dtype=int)

saps1 = np.zeros((len(eps),), dtype=int)
sofa  = np.zeros((len(eps),), dtype=int)
ym    = np.zeros((len(eps),), dtype=int)
ylos  = np.zeros((len(eps),), dtype=int)
ysurv = np.zeros((len(eps),), dtype=int)

for i, ep in enumerate(eps):
    x, ts = ep.as_nparray_with_timestamps()
    Xraw.append(x)
    tsraw.append(ts)
    Traw[i] = x.shape[0]

    x = ep.as_nparray_resampled(impute=False)
    Xmiss.append(x)
    T[i] = x.shape[0]

    x = ep.as_nparray_resampled(impute=True)
    X.append(x)

    recordid[i] = ep._recordId
    gender[i]   = ep._gender
    age[i]      = ep._age
    height[i]   = ep._height
    weight[i]   = ep._weight
    icutype[i]  = ep._icuType
    if ep._set == 'a':
        source[i] = 1
    elif ep._set == 'b':
        source[i] = 2
    elif ep._set == 'c':
        source[i] = 3
    else:
        source[i] = 0
    saps1[i] = ep._saps1
    sofa[i]  = ep._sofa
    ym[i]    = ep._mortality
    ylos[i]  = ep._los
    ysurv[i] = ep._survival

np.savez(os.path.join(args.out_dir, args.basename + '.npz'), Xraw=Xraw, tsraw=tsraw, Traw=Traw, Xmiss=Xmiss, X=X, T=T,
         recordid=recordid, gender=gender, age=age, height=height, weight=weight, icutype=icutype, source=source,
         saps1=saps1, sofa=sofa, ym=ym, ylos=ylos, ysurv=ysurv)

sio.savemat(os.path.join(args.out_dir, args.basename + '.mat'), {'Xraw': Xraw, 'tsraw': tsraw, 'Traw': Traw, 'Xmiss': Xmiss,
            'X': X, 'T': T, 'recordid': recordid, 'gender': gender, 'age': age, 'height': height, 'weight': weight,
            'icutype': icutype, 'source': source, 'saps1': saps1, 'sofa': sofa, 'ym': ym, 'ylos': ylos, 'ysurv': ysurv})

f = open(os.path.join(args.out_dir, args.basename + '-variables.csv'), 'w')
for i,v in enumerate(variables):
    f.write('{0},{1}\n'.format(i+1,v))
f.close()
