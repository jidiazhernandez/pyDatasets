# -*- coding: utf-8 -*-
"""
@author: dbell
@author: davekale

"""

import re
from datetime import timedelta, datetime

import numpy as np
from pandas import DataFrame, Series


class InvalidChallenge2012DataException(Exception):
    def __init__(self, field, err, value, recordid=None):
        s = '{0} has invalid {1}: {2}'.format(field, err, value)
        if recordid is not None:
            s = s + ' (record {0})'.format(recordid)
        Exception.__init__(self, s)
        self.field = field
        self.err = err
        self.value = value
        self.recordid = recordid

class Challenge2012Episode:
    def __init__(self, recordID, age, gender, height, icuType, weight, data, source_set='UNKNOWN'):
        """Constructor for Challenge2012Episode.

        :param recordID: Record ID
        :param age: patient age in years
        :param gender: patient gender (1: female, 0: male, np.nan: unknown/other)
        :param height: patient height in cm (-1: not available)
        :param icuType: type of ICU (1-4)
        :param weight: patient weight in kg (-1: not available)
        :param data: physiologic measurements as pandas.DataFrame object with
                datetime as index, one column per variable
        :param source_set: a, b, c, or UNKNOWN
        """
        self._recordId = recordID
        self._age      = age
        self._gender   = gender
        self._height   = height
        self._icuType  = icuType
        self._weight   = weight

        self._saps1     = -1
        self._sofa      = -1
        self._los       = -1
        self._survival  = -1
        self._mortality = -1

        self._data = data.copy()
        self._set = source_set

    @staticmethod    
    def generate_elapsed_timestamps(timestamps, first_dt):
        """Converts datetime to minutes elapsed since first_dt.

        Arguments:
        :param timestamps: pandas.Series containing datetime.datetime objects
        :param first_dt: datetime.datetime object
        """
        return (timestamps-first_dt).apply(lambda x: x / np.timedelta64(1, 'm'))

    @staticmethod
    def convert_time_str(s, today=None):
        """Converts time string in Challenge 2012 format to a datetime relative to "today."

        Challenge 2012 format is in MM:SS (minutes:seconds).

        Arguments:
        :param s: string in MM:SS format
        :param today: datetime.datetime object
        """
        m = re.match("(\d\d)\:(\d\d)", s)
        assert(m)
        hours = int(m.group(1))
        minutes = int(m.group(2))
        if today is None:
            today = datetime.today()
        return today + timedelta(hours=hours, minutes=minutes)

    @staticmethod
    def make_time_str_converter(today):
        """Closure function that returns a Challenge 2012 time string converter, relative to
        today argument.

        Arguments:
        :param today: datetime.datetime object
        """
        def converter(s):
            return Challenge2012Episode.convert_time_str(s, today=today)

        return converter

    @staticmethod
    def from_file(filename, variables):
        """Read data for one Challenge2012Episode from one text file, including only specified variables.

        Arguments:
        :param filename: string with full path to file to be read.
        :param variables: list of variable names to keep, as strings.
        """
        match = re.search('\d{6}.txt', filename) #ensure that file matches given format
        if not match:
            raise InvalidChallenge2012DataException('file', 'name', filename)
        df = DataFrame.from_csv(filename, index_col=None)
        df.drop_duplicates(subset=['Time', 'Parameter'], inplace=True)
        df = df.pivot(index='Time', columns='Parameter', values='Value')

        variables = set(variables)
        variables.update(['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight'])
        emptyvec = np.empty((df.shape[0],))
        emptyvec[:] = np.nan
        for v in variables:
            if v not in df.columns:
                df[v] = emptyvec
        for v in df.columns:
            if v not in variables:
                del df[v]

        df.sort(axis=1, inplace=True)
        df.reset_index(inplace=True)

        if df['RecordID'].notnull().sum() != 1:
            raise InvalidChallenge2012DataException('recordID', 'count', df['RecordID'].notnull().sum())
        recordId = int(df['RecordID'][df['RecordID'].first_valid_index()])
        del df['RecordID']

        if df['Age'].notnull().sum() != 1:
            raise InvalidChallenge2012DataException('Age', 'count', df['Age'].notnull().sum(), recordId)
        age = int(df['Age'][df['Age'].first_valid_index()])
        del df['Age']

        if df['Gender'].notnull().sum() != 1:
            raise InvalidChallenge2012DataException('Gender', 'count', df['Gender'].notnull().sum(), recordId)
        gender = 1 - int(df['Gender'][df['Gender'].first_valid_index()])
        gender = -1 if gender < 0 or gender > 1 else gender
        del df['Gender']

        if df['Height'].notnull().sum() != 1:
            raise InvalidChallenge2012DataException('Height', 'count', df['Height'].notnull().sum(), recordId)
        height = df['Height'][df['Height'].first_valid_index()]
        del df['Height']

        if df['ICUType'].notnull().sum() != 1:
            raise InvalidChallenge2012DataException('ICUType', 'count', df['ICUType'].notnull().sum(), recordId)
        icuType = int(df['ICUType'][df['ICUType'].first_valid_index()])
        if icuType not in {1,2,3,4}:
            raise InvalidChallenge2012DataException('ICUType', 'value', icuType, recordId)
        del df['ICUType']

        if df['Weight'].notnull().sum() < 1:
            raise InvalidChallenge2012DataException('Weight', 'count', df['Weight'].notnull().sum(), recordId)
        weight = df['Weight'][df['Weight'].first_valid_index()]

        df.replace(to_replace=-1, value=np.nan, inplace=True)

        df['TimeOriginal'] = df.Time
        converter = Challenge2012Episode.make_time_str_converter(datetime.today())
        try:
            df.Time = df.TimeOriginal.apply(converter)
        except:
            raise InvalidChallenge2012DataException('timestamp', 'format', df.TimeOriginal[0], recordId)
        del df['TimeOriginal']
        df.set_index('Time', inplace=True)
        df.sort(axis=1, inplace=True)
        df.sort(axis=0, inplace=True)

        m = re.search('/set-([abc])/\d{6}.txt', filename)
        if m:
            return Challenge2012Episode(recordId, age, gender, height, icuType, weight, df, source_set=m.group(1))
        else:
            return Challenge2012Episode(recordId, age, gender, height, icuType, weight, df)

    def printObj(self):
        print self.__dict__

    def merge_variables(self, to_merge):
        """Merges time series variables into new time series variables.

        :param to_merge: dictionary mapping new variable name to list of variables to be merged.
        :return:
        """
        dold = self._data.copy()
        s = Series(data=np.zeros((dold.shape[0],)), index=dold.index).replace(0, np.nan)
        dnew = DataFrame(dict([ (k, s) for k in to_merge.keys() if len(set(to_merge[k]).intersection(dold.columns))>0 ]))
        for newvar in dnew.columns:
            for oldvar in to_merge[newvar]:
                if oldvar in dold.columns:
                    dnew[newvar][dold[oldvar].notnull()] = dold[oldvar][dold[oldvar].notnull()]
                    del dold[oldvar]
        dnew = dnew.join(dold, how='outer')
        dnew.sort(axis=1, inplace=True)
        dnew.sort(axis=0, inplace=True)
        self._data = dnew
        
    def as_nparray_with_timestamps(self, hours=None):
        """Returns time series data as T x D matrix, along with T-vector of timestamps. T is number of samples, D is
        number of variables. Timestamps are in minutes elapsed and may be irregular.

        :param hours: trim data to maximum number of hours
        :return: tuple of TxD matrix of data, T vector of timestamps
        """
        df = self._data.reset_index()

        df['Elapsed'] = Challenge2012Episode.generate_elapsed_timestamps(df.Time, df.Time.min()).astype(int)
        if hours is not None:
            df = df.ix[df.Elapsed < hours*60]
        df.set_index('Elapsed', inplace=True)
        del df['Time']
        df.sort(axis=1, inplace=True)
        df.sort_index(inplace=True)
        return df.as_matrix(), df.index.to_series().as_matrix()
      
    def as_nparray_resampled(self, hours=None, rate='1H', bucket=True, impute=False): #, normal_values=None):
        """Returns time series data as resampled T x D matrix. T is number of samples, D is
        number of variables. Leverages pandas.DataFrame.resample routine. Can impute missing values for
        time series with at least one measurement.

        :param hours: trim data to maximum number of hours
        :param rate: target sampling rate (in string format, as required by pandas.DataFrame.resample
        :param bucket: if True, take mean of measurements in window; otherwise, use first measurement
        :param impute: if True, use forward- and backward-filling to impute missing measurements.
        :return: TxD matrix of data
        """
        df = self._data #.reset_index()

        if impute:
            df = df.resample(rate, how='mean' if bucket else 'first', closed='left', label='left', fill_method='ffill')
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)
            #        assert(df[varid].notnull().all())
        else:
            df = df.resample(rate, how='mean' if bucket else 'first', closed='left', label='left', fill_method=None)

        df.reset_index(inplace=True)
        df['Elapsed'] = Challenge2012Episode.generate_elapsed_timestamps(df.Time, df.Time.min()).astype(int)
        if hours is not None:
            df = df.ix[df.Elapsed < hours*60]
        df.set_index('Elapsed', inplace=True)
        del df['Time']
        df.sort(axis=1, inplace=True)
        df.sort_index(inplace=True)
        return df.as_matrix()

class ConflictingChallenge2012DataException(Exception):
    def __init__(self, field1, value1, field2, value2, recordid=None):
        s = '{0}, {1} values conflict: {2}, {3}'.format(field1, field2, value1, value2)
        if recordid is not None:
            s = s + ' (record {0})'.format(recordid)
        Exception.__init__(self, s)
        self.field1 = field1
        self.field2 = field2
        self.value1 = value1
        self.value2 = value2
        self.recordid = recordid

def add_outcomes(eps, outcomes_filename):
    """

    :param eps: list of Challenge2012Episode objects
    :param outcomes_filename: full path as string to outcomes CSV file
    :return: list of Challenge2012Episode objects with updated outcomes data
    """
    try:
        outcomes = DataFrame.from_csv(outcomes_filename, index_col='RecordID')
    except:
        raise InvalidChallenge2012DataException('outcome', 'filename', outcomes_filename)

    ## address conflicting outcomes data to make it consistent with
    ## rules described on http://physionet.org/challenge/2012/#data-correction
    #idx = (outcomes['In-hospital_death']==1)
    #idx = idx & (outcomes['Survival']<2) & (outcomes['Length_of_stay']>=2)
    #outcomes['Survival'][idx] = outcomes['Length_of_stay'][idx]
    #outcomes['Length_of_stay'][(outcomes['Length_of_stay']<2)&(outcomes['In-hospital_death']==0)] = -123456
    #outcomes['Survival'][(outcomes['Survival']>-1)&(outcomes['Survival']<2)] = -1
    ##
    outcomes.Survival[outcomes.Survival==-1] = -999
    outcomes.Survival[(outcomes.Survival==0)&(outcomes['In-hospital_death']==0)] = -999

    for ep in eps:
        if ep._recordId in outcomes.index:
            ep._saps1 = outcomes['SAPS-I'][ep._recordId]
            ep._sofa = outcomes['SOFA'][ep._recordId]
            ep._los = outcomes['Length_of_stay'][ep._recordId]
            ep._mortality = outcomes['In-hospital_death'][ep._recordId]
            if ep._mortality != 1 and ep._mortality != 0:
                raise InvalidChallenge2012DataException('mortality', 'value', ep._mortality, ep._recordId)
            ep._survival = outcomes['Survival'][ep._recordId]
            if (ep._survival > ep._los or ep._survival == -999) and ep._mortality != 0:
                raise ConflictingChallenge2012DataException('survival', ep._survival, 'mortality', ep._mortality, ep._recordId)
            if (ep._survival > 0 and ep._survival <= ep._los) and ep._mortality != 1:
                raise ConflictingChallenge2012DataException('survival', ep._survival, 'mortality', ep._mortality, ep._recordId)
            
    return eps
