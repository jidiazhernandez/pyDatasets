# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 11:24:22 2015

@author: davekale
"""

import argparse
import csv
import glob
import numpy as np
import os
import scipy.io as sio
import sys

# make sure set PYTHONPATH environment variable or
# modify sys.path to be able to find pydatasets package
from pydatasets.uci.eeg import EegSubject

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=unicode)
parser.add_argument('out_dir', type=unicode)
args = parser.parse_args()

try:
    os.makedirs(args.out_dir)
except:
    pass    
    
#try:
#    os.makedirs(os.path.join(out_dir, 'by_subject'))
#except:
#    pass
#try:
#    os.makedirs(os.path.join(out_dir, 'by_trial'))
#except:
#    pass

# this is the number of trial files
N = len(glob.glob(os.path.join(args.data_dir, 'co*/co*.rd.*.gz')))

# D=64 is the number of channels (time series dimensionality)
# T=256 is the number of samples (number of timesteps in time series)
# N is the number of trials (each trial is a multivariate time series or "instance")
X = np.zeros((N, 256, 64))
channel_names = []
channel_numbers = []

# info is for storing:
# 0: instance index: 0, ..., N-1
# 1: length of time series
# 2: alcoholic: 1 if a, -1 if c
# 3: subject ID code: 0, ..., 121
# 4: trial number: 0, ..., 119
# 5: stimulus type code: 0, 1, 2, 3, 4
T           = np.zeros((N,), dtype=int)
subjectid   = np.zeros((N,), dtype=int)
trialnum    = np.zeros((N,), dtype=int)
y_alcoholic = np.zeros((N,), dtype=int)
y_stimulus  = np.zeros((N,), dtype=int)

# for coding stimulus types
stimuli = dict()
stim_next = 1
# for coding subjects
sids = dict()
sid_next = 1
# index for instances
idx = 0

# iterate over subjects (directories in "eeg_full")
for dirname in glob.glob(os.path.join(args.data_dir, 'co*')):
    if os.path.isdir(dirname):
        # read in all of a subject's data
        s = EegSubject.from_directory(dirname)
    
        # itereate over this subject's trials
        sys.stdout.write('Processing trials')
        for trial in s.trials_data:
            sys.stdout.write('.')
       
            # code stimulus
            if trial.stimulus not in stimuli:
                stimuli[trial.stimulus] = stim_next
                stim_next += 1
            stim_num = stimuli[trial.stimulus]
            
            # code subject
            if trial.subject_id not in sids:
                sids[trial.subject_id] = sid_next
                sid_next += 1
            sid = sids[trial.subject_id]
        
            # store info
            T[idx]           = 256
            subjectid[idx]   = sid
            trialnum[idx]    = trial.trial_num
            y_alcoholic[idx] = 1 if trial.alcoholic else 0
            y_stimulus[idx]  = stim_num
            
            # store matrix form of trial time series
            X[idx,:,:] = trial.as_nparray()
            cn = [str(idx)]
            cn.extend(trial.get_channel_names())
            channel_names.append(cn)
            cn = [str(idx)]
            cn.extend([ str(n) for n in trial.get_channel_numbers() ])
            channel_numbers.append(cn)
            
            # code instance
            idx += 1
        sys.stdout.write('DONE!\n')

X           = X[:,:,0:idx]
T           = T[0:idx]
subjectid   = subjectid[0:idx]
trialnum    = trialnum[0:idx]
y_alcoholic = y_alcoholic[0:idx]
y_stimulus  = y_stimulus[0:idx]

#cts = np.reshape(((X==0).sum(1)==256).sum(0), (X.shape[2],1))

# Store X, info numpy arrays to disk
sio.savemat(os.path.join(args.out_dir, 'ucieeg.mat'),
            {'X': X, 'T': T, 'subjectid': subjectid, 'trialnum': trialnum,
             'y_alcoholic': y_alcoholic, 'y_stimulus': y_stimulus})
np.savez(os.path.join(args.out_dir, 'ucieeg.npz'),
         X=X, subjectid=subjectid, trialnum=trialnum,
         y_alcoholic=y_alcoholic, y_stimulus=y_stimulus)

with open(os.path.join(args.out_dir, 'ucieeg-channel_names.csv'), 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for cn in channel_names:
        writer.writerow(cn)

with open(os.path.join(args.out_dir, 'ucieeg-channel_numbers.csv'), 'wb') as csvfile:
    writer = csv.writer(csvfile)
    for cn in channel_numbers:
        writer.writerow(cn)

with open(os.path.join(args.out_dir, 'ucieeg-readme.txt'), 'w') as infofile:
    infofile.write("""INFO ABOUT PROCESSED UCI EEG DATA SET

"series" is a C x T x N tensor of EEG measurements where C=64 is the number of
channels, T=256 is the number of samples (same for every) time series, and N is
the number of trials.

"T" is a N-vector of time series lengths (all equal to 256).

"subjectid" is a N-vector of subjectids since we have ~100 time series for each
of the ~120 subjects in the study.

"trialnum" is a N-vector of per-subject trial numbers. Each subject had ~120
trials in the original study but not in this data set.

"y_alcoholic" is a N-vector of binary labels, 1 if the subject is an alcoholic,
0 if the subject is not.

"y_stimulus" is a N-vector of categorical labels indicating which stimulus was
shown to the subject during the trial. In the study, there were five different
stimuli used, so the labels range from 1 to 5, indicating the following:
""")
    for k,v in sorted(stimuli.iteritems()):
        infofile.write('\t{0}: {1}\n'.format(v, k))


## Store subject ID codes to CSV
#f = open(os.path.join(out_dir, 'subjectids.csv'), 'w')
#for k, v in sorted(sids.iteritems(), key=itemgetter(1, 0)):
#    f.write('{0},{1}\n'.format(v, k))
#f.close()

## Store simutulus codes to CSV
#f = open(os.path.join(out_dir, 'stimuli.csv'), 'w')
#for k, v in sorted(stimuli.iteritems(), key=itemgetter(1, 0)):
#    f.write('{0},{1}\n'.format(v, k))
#f.close()
