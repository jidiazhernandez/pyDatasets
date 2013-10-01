from __future__ import division

import numpy as np
import os
import pandas as pd
import re

from datetime import datetime, timedelta
from pandas import DataFrame, Series

class InvalidIcuDataException(Exception):
	def __init__(self, field, err, value):
		Exception.__init__(self, '{0} has invalid {1}: {2}'.format(field, err, value))
		self.field = field
		self.err = err
		self.value = value

class IcuEpisode:
    def __init__(self, episodeid, sex, weight, dob, admit_dt, discharge_dt, died, prim_diag, msmts, sec_diag=[], med_disch_dt=None,
                 admit_pcpc=np.nan, discharge_pcpc=np.nan, admit_popc=np.nan, discharge_popc=np.nan):
        self.episodeid = episodeid
        self.sex = sex
        self.weight = float(weight)

        #assert(type(dob) == datetime)
        if type(dob) != datetime:
        	raise InvalidIcuDataException('dob', 'type', type(dob))
        self.dob = dob

        #assert(type(admit_dt) == datetime)
        if type(admit_dt) != datetime:
        	raise InvalidIcuDataException('admit_dt', 'type', type(admit_dt))
        self.admit_dt = admit_dt
        
        #assert(type(discharge_dt) == datetime)
        if type(discharge_dt) != datetime:
        	raise InvalidIcuDataException('discharge_dt', 'type', type(discharge_dt))
        self.discharge_dt = discharge_dt
        
        #assert(type(died) == bool)
        if type(died) != bool:
        	raise InvalidIcuDataException('died', 'type', type(died))
        self.died = died
        
        if type(prim_diag) == int:
        	self.prim_diag = str(prim_diag)
        elif type(prim_diag) == str:
        	if not re.match('^\d+$', prim_diag):
        		raise InvalidIcuDataException('died', 'value', died)
        	self.prim_diag = prim_diag
        else:
        	raise InvalidIcuDataException('prim_diag', 'type', type(died))
        
        self.sec_diag = list(sec_diag)
        
        #assert(med_disch_dt is None or type(med_disch_dt) == datetime)
        if med_disch_dt is not None and type(med_disch_dt) != datetime:
			raise InvalidIcuDataException('med_disch_dt', 'type', type(med_disch_dt))
        self.med_disch_dt = med_disch_dt

        self.admit_pcpc = float(admit_pcpc)
        self.discharge_pcpc = float(discharge_pcpc)
        self.admit_popc = float(admit_popc)
        self.discharge_popc = float(discharge_popc)
        
        #assert(type(msmts) == DataFrame)
        if type(msmts) != DataFrame:
        	raise InvalidIcuDataException('msmts', 'type', type(msmts))
        elif msmts.shape[0] == 0:
        	raise InvalidIcuDataException('msmts', 'size', msmts.shape[0])
        self.msmts = msmts

        self.first_dt = msmts.index.min()

    def as_nparray(self, hours=None):
        """Spits out data as a D x T numpy.array (D=# channels, T=# samples)

        Notes:
        Notice what we do here: we start with a pandas.DataFrame where each channel
        is a column (so you can think of it as a T x D matrix). We first rename the
        columns to channel numbers,then sort the columns, then sort the index, then
        transform to numpy.array, then finally take the transpose to get D x T.
        rename(columns=self.wafer_id)
        """
        if hours is None:
	        df = self.msmts.sort(axis=1).sort_index()
        else:
        	#df = self.msmts.ix[self.msmts.index < self.admit_dt + timedelta(hours=float(hours))].sort(axis=1).sort_index()
        	df = self.msmts.ix[self.msmts.index < self.first_dt + timedelta(hours=float(hours))].sort(axis=1).sort_index()
        #timestamps = np.array((Series(df.index)-self.admit_dt).apply(lambda x: x / np.timedelta64(1, 'm')))
        timestamps = np.array((Series(df.index)-self.first_dt).apply(lambda x: x / np.timedelta64(1, 'm')))
        return timestamps, df.as_matrix().T

    def as_nparray_resampled(self, hours=None, rate='1H', bucket=True):
        """Spits out data as a D x T numpy.array (D=# channels, T=# samples)

        Notes:
        Notice what we do here: we start with a pandas.DataFrame where each channel
        is a column (so you can think of it as a T x D matrix). We first rename the
        columns to channel numbers,then sort the columns, then sort the index, then
        transform to numpy.array, then finally take the transpose to get D x T.
        rename(columns=self.wafer_id)
        """
        if hours is None:
	        df = self.msmts
        else:
        	#df = self.msmts.ix[self.msmts.index < self.admit_dt + timedelta(hours=float(hours))]
        	df = self.msmts.ix[self.msmts.index < self.first_dt + timedelta(hours=float(hours))]
        df = df.resample(rate, how='mean' if bucket else 'first', fill_method='pad').sort(axis=1).sort_index()
        if df.shape[0] == 0:
        	print self.episodeid, self.msmts.shape, df.shape
        	assert(False)
        for c in df:
			if c not in df.columns:
				df[c] = np.nan
			if np.isnan(df[c].ix[-1]):
				df[c].ix[-1] = 0
			if df[c].notnull().sum() > 0:
				df[c].ix[df[c].isnull()] = df[c].ix[df[c].notnull()][0]
			df[c] = df[c] - df[c].mean()
        #timestamps = np.array((Series(df.index)-self.admit_dt).apply(lambda x: x / np.timedelta64(1, 'm')))
        timestamps = np.array((Series(df.index)-self.first_dt).apply(lambda x: x / np.timedelta64(1, 'm')))
        return timestamps, df.ix[0:min(24, df.shape[0])].as_matrix().T, df.ix[0:min(24, df.shape[0])]

    def age_in_months(self):
    	return int(np.floor((self.admit_dt - self.dob).total_seconds() / (60 * 60 * 24 * 365) * 12))

    def delta_pcpc(self):
    	return self.discharge_pcpc - self.admit_pcpc

    def delta_popc(self):
    	return self.discharge_popc - self.admit_popc

class OldVpicuEpisode(IcuEpisode):

	@staticmethod
	def from_directory(path):
		episodeid = int(re.match('.*/(\d+)', path).group(1))
		sf = pd.read_csv(os.path.join(path, 'static.csv'), index_col=None, parse_dates=[2, 6, 7, 9], na_values=['null']).T.squeeze()
		sex = np.nan if type(sf['SEX'])!=str else (1 if sf['SEX']=='f' else (0 if sf['SEX']=='m' else np.nan))
		weight = float(sf['WEIGHT'])
		dob = sf['DOB']
		admit = sf['ADMIT']
		discharge = sf['DISCHARGE']
		mdischarge = None if type(sf['MEDICALDISCHARGE']) != datetime else sf['MEDICALDISCHARGE']
		died = None if np.isnan(sf['DIED']) else (int(sf['DIED']) > 0)

		cf = pd.read_csv(os.path.join(path, 'codes.csv'), index_col=None, na_values=['null']).T.squeeze()
		apcpc = float(cf['ADMITPCPC'])
		apopc = float(cf['ADMITPOPC'])
		dpcpc = float(cf['DISCHARGEPCPC'])
		dpopc = float(cf['DISCHARGEPOPC'])

		df = pd.read_csv(os.path.join(path, 'diagnoses.csv'), index_col=None, na_values=['null'])
		#assert(df.shape[0]>0 and df.PRIMARY.sum()==1)
		#if df.shape[0] == 0 or df.PRIMARY.sum() > 1:
		#	raise InvalidIcuDataException('prim_diag', 'size', df.PRIMARY.sum())
		sdiags = [ '{0:04}'.format(code) for code in set(df.CODE[df.PRIMARY<1].tolist()) ]
		if df.PRIMARY.sum() == 1:
			pdiag = '{0:04}'.format(df.CODE[df.PRIMARY>0][0])
		elif df.PRIMARY.sum() > 1:
			pdiag = '2222'
		else:
			pdiag = '0000'

		def convert_value(s):
		    s = re.sub('[^\.0123456789]', '', s)
		    if s != '':
		        return float(s)
		    else:
		        return np.nan

		pf = pd.read_csv(os.path.join(path, 'variables.csv'), index_col=None, parse_dates=[0], na_values=['null'], converters={ 'VALUE': convert_value })
		pf = pf.sort(columns=['TIME', 'VARID'])
		pf = pf.pivot(index='TIME', columns='VARID', values='VALUE')
		for c in range(1, 14):
			if c not in pf.columns:
				pf[c] = np.nan

		return IcuEpisode(episodeid, sex, weight, dob, admit, discharge, died, pdiag, pf, sec_diag=sdiags, med_disch_dt=mdischarge,
		        		  admit_pcpc=apcpc, discharge_pcpc=dpcpc, admit_popc=apopc, discharge_popc=dpopc)

