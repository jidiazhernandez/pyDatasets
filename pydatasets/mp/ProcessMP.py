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
import time
import copy
import csv

from pandas import Panel, DataFrame, Series
import MP

_NOTEBOOK = False; # change to true if you're going to run this locally 

parser = argparse.ArgumentParser()
parser.add_argument('data', type=unicode)
parser.add_argument('out', type=unicode)
parser.add_argument('outcomes', type=unicode)
#parser.add_argument('-vi', '--var_info', type=unicode, default='../../data/vpicu/physiologic-variable-normals.csv') # I need to look at this
parser.add_argument('-rr', '--resample_rate', type=int, default=60)
parser.add_argument('-ma', '--max_age', type=int, default=18)

if _NOTEBOOK:
    args = parser.parse_args( 'settings go here' ) #you're going to have to update with proper params
else:
    args = parser.parse_args()
    
data_dir = args.data
out_dir = args.out
outcomes_dir = args.outcomes
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
allPatientObjs = dict()

#Traw = np.zeros((N,), dtype=int)    # Number of (irregular) samples for episode
#T = np.zeros((N,), dtype=int)       # Resampled episode lengths (for convenience)
#age = np.zeros((N,), dtype=int)     # Per-episode patient ages in months
#gender = np.zeros((N,), dtype=int) # Per-episode patient gender
#weight = np.zeros((N,))             # Per-episode patient weight
#lmf = np.zeros((N,))                # Last-first duration

allCondensedVals = None
channels = 'channels.txt'
channelsDict = dict()

num = 0
chans = file(channels, 'r')
for chan in chans:
    chan = chan.rstrip()
    channelsDict[chan] = {}
    num = num + 1
chans.close()

data_dict = {'sampling_rate': {'mean': [], 'standard_dev': None }, 'value': {'mean': [], 'standard_dev': None }, 'missing': 0 } 

meta_statics = copy.deepcopy(data_dict)

patient_stats = copy.deepcopy(channelsDict)
overall_stats = copy.deepcopy(channelsDict)
for key in channelsDict:
    patient_stats[key] = copy.deepcopy(data_dict)
    overall_stats[key] = copy.deepcopy(data_dict)

for key in channelsDict:
    channelsDict[key] = []

idx = 0
count_nodata = 0
total_time_start = time.time()
for pat in patients:
    start_create = time.time()
    try:
        subj = MP.MPSubject.from_file(os.path.join(data_dir, pat), channelsDict)
    except MP.InvalidMPDataException as e:
        count_nodata += 1
        if e.field == 'msmts' and e.err == 'size':
            pass
        else:
            sys.stdout.write('\nskipping: ' + str(e))
        continue
    end_create = time.time()
    allPatientObjs[subj._recordID] = subj
    print "Time to create: " 
    print (end_create - start_create)
    start_post = time.time()
    if allCondensedVals is None: 
        allCondensedVals = subj.condensed_values()
    else:
        condVals = subj.condensed_values()
        for feature in allCondensedVals:
            if condVals[feature]:
                patient_stats[feature]['value']['mean'].append( np.mean(condVals[feature]) )
                patient_stats[feature]['sampling_rate']['mean'].append( len(condVals[feature]) )
                
                overall_stats[feature]['value']['mean'] += condVals[feature]
                overall_stats[feature]['sampling_rate']['mean'].append( len(condVals[feature]) )
                
                allCondensedVals[feature] += (condVals[feature])
            else:
                patient_stats[feature]['missing'] = 1
                overall_stats[feature]['missing'] += 1
                
    s = subj.as_nparray()
    if s.size == 0: 
        print 'getting skipped'
        continue
    mylmf  = (s[:,0].max() - s[:,0].min())/60
    
    #store raw time series
    Xraw.append(s)
    
    #resample
    sr = subj.as_nparray_resampled(rate=resample_rate, impute=True)
    #first couple resampled have nan vals
    if not np.all(~np.isnan(s)):
        print pat
    X.append(sr)
    
    
    idx += 1
    end_post = time.time()
    print "Time for post logic: " 
    print (end_post - start_post)


for feature in allCondensedVals:
    overall_stats[feature]['value']['standard_dev'] = np.std(overall_stats[feature]['value']['mean'])
    overall_stats[feature]['value']['mean'] = np.mean(overall_stats[feature]['value']['mean'])
    overall_stats[feature]['sampling_rate']['standard_dev'] = np.std(overall_stats[feature]['sampling_rate']['mean'])
    overall_stats[feature]['sampling_rate']['mean'] = np.mean(overall_stats[feature]['sampling_rate']['mean'])
    
    patient_stats[feature]['value']['standard_dev'] = np.std(patient_stats[feature]['value']['mean'])
    patient_stats[feature]['value']['mean'] = np.mean(patient_stats[feature]['value']['mean'])
    patient_stats[feature]['sampling_rate']['standard_dev'] = np.std(patient_stats[feature]['sampling_rate']['mean'])
    patient_stats[feature]['sampling_rate']['mean'] = np.mean(patient_stats[feature]['sampling_rate']['mean'])
    
    overall_stats[feature]['missing'] = overall_stats[feature]['missing'] / float(idx - count_nodata)
    patient_stats[feature]['missing'] = patient_stats[feature]['missing'] / float(idx - count_nodata)

print 'overall'        
print overall_stats
print 'patient'     
print patient_stats    
total_time_end = time.time()
#print allCondensedVals 
print "Total elapsed time: "
print (total_time_end - total_time_start) 

#outcome code
outcomeFile = file(outcomes_dir, 'r')
outcomeFile.readline()
numPatients = 0
deathCount = 0
no_outcome = 0
length_stays = []
for line in csv.reader(outcomeFile, delimiter=','):
    numPatients += 1
    rID = int(line[0])
    if rID in allPatientObjs:
        patObj = allPatientObjs[rID]
        death = int(line[5])
        patObj.death = death
        deathCount += death
        los = int(line[3])
        patObj.length_of_stay = los
        length_stays.append(los)
        if hasattr(patObj, '_condValues'):
            del patObj._condValues
        patObj.to_pickle('/Users/dbell/Desktop/pickled_data')
    else:
        no_outcome += 1
deathPercentage = float(deathCount) / float(numPatients)
mean_length_stay = np.mean(length_stays)
standard_dev_length_stay = np.std(length_stays)
print 'death percentage: '
print deathPercentage
print 'No data: '
print count_nodata
print 'No outcome: '
print no_outcome
    
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
    
    
    
    