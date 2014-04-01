# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
import os 
import pandas as pd 
import re 
import csv

from pandas import DataFrame, Series
from datetime import datetime, timedelta

class InvalidMPDataException(Exception):
    def __init__(self, field, err, value):
        Exception.__init__(self, '{0} has invalid {1}: {2}'.format(field, err, value))
        self.field = field
        self.err = err
        self.value = value

class MPSubject:
    def __init__(self, recordID, age, gender, height, icuType, weight, ts):
        self._recordID = recordID
        self._age = age
        self._gender = gender
        self._height = height
        self._icuType = icuType
        self._weight = weight
        self._data = ts.copy()
        print self.as_nparray()
        
    @staticmethod
    def from_file(filename, channels='channels.txt'):
        match = re.match('.*\/\d\d\d\d\d\d.txt', filename) #ensure that file matches given format
        if not match:
            raise InvalidMPDataException('file', 'name', filename)
        openFile = file(filename, 'r') #open patient's text file
        tsmap = dict() #dictionary to map channel names to list of lists for time series
        
        #use channels text file to create channels in dict
        chans = file(channels, 'r')
        for chan in chans:
            chan = chan.rstrip()
            tsmap[chan] = []
            tsmap[chan].append([])
            tsmap[chan].append([])
        chans.close()
        
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
                    raise InvalidMPDataException() #need args
                hours = int(tm.group(1))
                minutes = int(tm.group(2))
                days = int(hours / 24)
                hours = hours % 24
                tm = timedelta(days=days, hours=hours, minutes=minutes)
                #handle value entry into channel
                channel = line[1]
                try:
                    alreadyPresent = False 
                    for tmd in tsmap[channel][0]:
                        if tmd == tm:
                            alreadyPresent = True
                    if not alreadyPresent:
                            tsmap[channel][0].append(tm)
                            tsmap[channel][1].append(float(line[2]))
                except KeyError:
                    raise InvalidMPDataException('channel', 'name', channel) 
                    
        #insert the weight general descriptor as the inital entry for the weight channel            
        tsmap['Weight'][0].insert(0, timedelta(hours=0, minutes=0))
        tsmap['Weight'][1].insert(0, float(gnrlDescript[5]))
        
        #replace lists of lists of timeseries with Series in channel entries
        for key in tsmap:
            tsmap[key] = Series(data=tsmap[key][1], index=tsmap[key][0], name=key)
        dat = DataFrame.from_dict(tsmap)

        return MPSubject(int(gnrlDescript[0]), int(gnrlDescript[1]), bool(gnrlDescript[2]), float(gnrlDescript[3]), int(gnrlDescript[4]), float(gnrlDescript[5]), dat)
        
    def as_nparray(self):
        return self._data.sort(axis=1).sort_index().reset_index().as_matrix()
      
    
        
    
        
        
        
                    
                
                    
                
                
            
            
            
            
            
            
            
            
            
            
            
        
        
        
            
            
        
        
        
        
        
        