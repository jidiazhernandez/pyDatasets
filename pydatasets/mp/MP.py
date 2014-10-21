# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import os 
import pandas as pd 
import re 
import csv
import pickle
import copy

from pandas import DataFrame, Series
from datetime import timedelta

class InvalidMPDataException(Exception):
    def __init__(self, field, err, value):
        Exception.__init__(self, '{0} has invalid {1}: {2}'.format(field, err, value))
        self.field = field
        self.err = err
        self.value = value

class MPSubject:
    def __init__(self, recordID, age, gender, height, icuType, weight, ts, condValues):
        self._recordID = recordID
        self._age = age
        self._gender = gender
        self._height = height
        self._icuType = icuType
        self._weight = weight
        self._data = ts.copy()
        self._condValues = condValues.copy()
        #print self._data
        #print self.as_nparray()
        
        
    @staticmethod
    def from_file(filename, channelsDict):
        match = re.match('.*\/\d\d\d\d\d\d.txt', filename) #ensure that file matches given format
        if not match:
            raise InvalidMPDataException('file', 'name', filename)
        openFile = file(filename, 'r') #open patient's text file
        tsmap = copy.deepcopy(channelsDict) #dictionary to map channel names to list of lists for time series
        for key in tsmap:
            tsmap[key].append([])
            tsmap[key].append([])
        featureMap = copy.deepcopy(channelsDict) #dict to hold a condensed list of features

        #add times series data to existing map
        openFile.readline()
        attrCount = 0 #count of general descriptors read
        gnrlDescript = [] #list for general descriptors
        for line in csv.reader(openFile, delimiter=','):
            if attrCount < 6:
                gnrlDescript.append(line[2])
                attrCount = attrCount + 1
            else:
                #handle date
                tm = re.match("(\d\d)\:(\d\d)", line[0])
                if not match:
                    print line[0] #need args
                hours = int(tm.group(1))
                minutes = int(tm.group(2))
                days = int(hours / 24)
                hours = hours % 24
                tm = days*24*60 + hours*60 + minutes
                if tm < 0:
                    print "Negative time"
                #handle value entry into channel
                channel = line[1]
                try:
                    alreadyPresent = False 
                    for tmd in tsmap[channel][0]:
                        if tmd == tm:
                            alreadyPresent = True
                    if not alreadyPresent:
                        tsmap[channel][0].append(tm)
                        val = float(line[2])
                        tsmap[channel][1].append(val)
                        featureMap[channel].append(val)
                except KeyError:
                    raise InvalidMPDataException('channel', 'name', channel) 
                    
        #insert the weight general descriptor as the inital entry for the weight channel            
        #tsmap['Weight'][0].insert(0, 0)
        #tsmap['Weight'][1].insert(0, float(gnrlDescript[5]))
        
        #replace lists of lists of timeseries with Series in channel entries
        for key in tsmap:
            tsmap[key] = Series(data=tsmap[key][1], index=tsmap[key][0], name=key)
        dat = DataFrame.from_dict(tsmap)

        return MPSubject(round(float(gnrlDescript[0])), round(float(gnrlDescript[1])), bool(gnrlDescript[2]), float(gnrlDescript[3]), round(float(gnrlDescript[4])), float(gnrlDescript[5]), dat, featureMap)
        
    def condensed_values(self):
         return self._condValues   
         
    def get_data(self):
        return self._data.copy()
        
    def printObj(self):
        print self.__dict__
        
    def as_nparray(self):
        df = self._data.copy()
        return df.sort(axis=1).sort_index().reset_index().as_matrix() #I think you have to transpose to get D x T array
      
    def as_nparray_resampled(self, hours=None, rate='1H', bucket=True, impute=False, normal_values=None):
        df = self._data.copy()
        df.sort(axis=1, inplace=True)
        df.sort_index(inplace=True)
        
        if impute:
            #df = df.resample(rate, how='mean' if bucket else 'first', closed='left', label='left', fill_method='ffill')
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)
        else:
            df = df.resample(rate, how='mean' if bucket else 'first', closed='left', label='left', fill_method=None)
        
        df.reset_index(inplace=True)
        return df.as_matrix()
        
    def to_pickle(self, path='', filename=None):
        """
        Keyword arguments:
        path -- path without filename to store pickled object
        filename -- filename to append to path, optional
        """

        if filename is None:
            filename = 'MPSubject-{0._recordID}.pkl'.format(self)
        filename = os.path.join(path, filename)
        pickle.dump(self, open(filename, 'w'), pickle.HIGHEST_PROTOCOL)
        
    @staticmethod
    def from_pickle(filename):
        """Create MPSubject object from pickle file."""

        return pickle.load(open(filename, 'r'))

    def to_csv(self, path='', filename_base=None):
        """
        Keyword arguments:
        path -- directory to store CSVs
        filename_base -- base of filename (extensions will be appended)
        """
    
        if filename_base is None:
            filename_base = 'MPSubject-{0._recordID}'.format(self)
            filename_base = os.path.join(path, filename_base)
            temp = self._data.rename(columns=self._channels).sort(axis=1).sort_index()
            temp.to_csv(filename_base + '.data', header=False, index=False)
            f = open(filename_base + '.info', 'w')
            f.write('subject_id,{0._recordID}\nage,{0._age}\ngender,{1}\nheight,{0._height}\nICUType,{0._icuType}\nweight,{0._weight}'.format(self, 'male' if self._gender else 'female'))
            f.close()
        