from __future__ import division

import numpy as np
import os
import pandas as pd
import re

from datetime import datetime, timedelta
from pandas import DataFrame, Series

class InvalidVpicuDataException(Exception):
    def __init__(self, field, err, value):
        Exception.__init__(self, '{0} has invalid {1}: {2}'.format(field, err, value))
        self.field = field
        self.err = err
        self.value = value

class VpicuEpisode:
    def __init__(self, episodeid, sex, weight, dob, admit_dt, discharge_dt, died, prim_diag, msmts, diag=[], med_disch_dt=None,
                 admit_pcpc=np.nan, discharge_pcpc=np.nan, admit_popc=np.nan, discharge_popc=np.nan):
        self.episodeid = episodeid
        self.sex = sex
        self.weight = float(weight)

        #assert(type(dob) == datetime)
        if type(dob) != datetime:
            raise InvalidVpicuDataException('dob', 'type', type(dob))
        self.dob = dob

        #assert(type(admit_dt) == datetime)
        if type(admit_dt) != datetime:
            raise InvalidVpicuDataException('admit_dt', 'type', type(admit_dt))
        self.admit_dt = admit_dt
        
        #assert(type(discharge_dt) == datetime)
        if type(discharge_dt) != datetime:
            raise InvalidVpicuDataException('discharge_dt', 'type', type(discharge_dt))
        self.discharge_dt = discharge_dt
        
        #assert(type(died) == bool)
        if type(died) != bool:
            raise InvalidVpicuDataException('died', 'type', type(died))
        self.died = died

        if type(prim_diag) == str and not re.match('^\d+$', prim_diag):
                raise InvalidVpicuDataException('prim_diag', 'value', died)

        self.prim_diag = int(prim_diag)
        self.diag = set([ int(d) for d in diag ])
        self.diag.add(self.prim_diag)
        
        #assert(med_disch_dt is None or type(med_disch_dt) == datetime)
        if med_disch_dt is not None and type(med_disch_dt) != datetime:
            raise InvalidVpicuDataException('med_disch_dt', 'type', type(med_disch_dt))
        self.med_disch_dt = med_disch_dt

        self.admit_pcpc = float(admit_pcpc)
        self.discharge_pcpc = float(discharge_pcpc)
        self.admit_popc = float(admit_popc)
        self.discharge_popc = float(discharge_popc)
        
        #assert(type(msmts) == DataFrame)
        if type(msmts) != DataFrame:
            raise InvalidVpicuDataException('msmts', 'type', type(msmts))
        elif msmts.shape[0] == 0:
            raise InvalidVpicuDataException('msmts', 'size', msmts.shape[0])
        self.msmts = msmts
        self.msmts.sort(axis=1, inplace=True)
        self.msmts.sort_index(inplace=True)
        self.first_dt = self.msmts.index.min()

    @staticmethod
    def generate_timestamps(df, first_dt):
        return (df.index.to_series()-first_dt).apply(lambda x: x / np.timedelta64(1, 'm'))

    def adjust_msmts_for_age_gender(self, ranges):
        for varid in ranges.items:
            row        = ranges[varid].ix[((ranges[varid].Gender==self.sex) | (ranges[varid].Gender.isnull())) & (ranges[varid].AgeLow<=self.age_in_months()) & (ranges[varid].AgeHigh>self.age_in_months())]
            assert(row.shape[0]==1)
            row_target = ranges[varid].ix[((ranges[varid].Gender==self.sex) | (ranges[varid].Gender.isnull())) & (ranges[varid].AgeLow<=ranges[varid].AgeLow.max()) & (ranges[varid].AgeHigh>ranges[varid].AgeLow.max())]
            assert(row_target.shape[0]==1)
            self.msmts[varid] = (self.msmts[varid] - float(row.Low)) / float(row.High-row.Low) * float(row_target.High-row_target.Low) + float(row_target.Low)

    def extract_features(self, normal_values = None):
        if normal_values is None:
            normal_values = np.zeros((self.msmts.shape[1],))

        feats = self.msmts.describe()
        feats.ix[2][feats.ix[2].isnull()] = 0
        for varid in feats.columns:
            feats[varid][feats[varid].isnull()] = normal_values[varid]
        feats = feats.ix[1:].as_matrix()

        trend_info = np.zeros((2, self.msmts.columns.shape[0]))
        for varid in self.msmts.columns:
            vals = self.msmts[varid][self.msmts[varid].notnull()]
            if vals.shape[0] <= 1:
                trend_info[0,varid-1] = 0
                if vals.shape[0] == 0:
                    trend_info[1,varid-1] = normal_values[varid]
                else:
                    trend_info[1,varid-1] = vals[0]
            else:
                trend_info[:,varid-1] = np.polyfit((vals.index.to_series() - vals.index.min()).apply(lambda x: x / np.timedelta64(1, 'm')), vals, 1)
        
        return np.vstack([ feats, trend_info ])
    
    def as_nparray(self, hours=None):
        """Spits out data as a D x T numpy.array (D=# channels, T=# samples)

        Notes:
        Notice what we do here: we start with a pandas.DataFrame where each channel
        is a column (so you can think of it as a T x D matrix). We first rename the
        columns to channel numbers,then sort the columns, then sort the index, then
        transform to numpy.array, then finally take the transpose to get D x T.
        rename(columns=self.wafer_id)
        """
        df = self.msmts.copy()
        df['ELAPSED'] = VpicuEpisode.generate_timestamps(df, self.first_dt)
        df.set_index('ELAPSED', inplace=True)
        if hours is not None:
            df = df.ix[df.index < hours*60]
        df.sort(axis=1, inplace=True)
        df.sort_index(inplace=True)
        df.reset_index(inplace=True)
        return df.as_matrix()

    def as_nparray_resampled(self, hours=None, rate='1H', bucket=True, impute=False, normal_values=None):
        """Spits out data as a D x T numpy.array (D=# channels, T=# samples)

        Notes:
        Notice what we do here: we start with a pandas.DataFrame where each channel
        is a column (so you can think of it as a T x D matrix). We first rename the
        columns to channel numbers,then sort the columns, then sort the index, then
        transform to numpy.array, then finally take the transpose to get D x T.
        rename(columns=self.wafer_id)
        """
        df = self.msmts.copy()
        df['TIME'] = df.index-timedelta(minutes=self.first_dt.minute)
        df.set_index('TIME', inplace=True)

        elapsed = VpicuEpisode.generate_timestamps(df, df.index.min())
        if hours is not None:
            df = df.ix[elapsed < hours*60]
        df.sort(axis=1, inplace=True)
        df.sort_index(inplace=True)

        if impute:
            for c in df.columns:
                fix = df.index[0]
                vix = np.where(df[c].notnull())[0]
                vix = vix[0] if vix.shape[0] > 0 else None
                if np.isnan(df[c].ix[fix]) and vix is not None:
                    df[c].ix[fix] = df[c].ix[vix]
            df = df.resample(rate, how='mean' if bucket else 'first', closed='left', label='left', fill_method='ffill')
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)
        else:
            df = df.resample(rate, how='mean' if bucket else 'first', closed='left', label='left', fill_method=None)

        df['ELAPSED'] = VpicuEpisode.generate_timestamps(df, df.index.min())
        df.set_index('ELAPSED', inplace=True)

        if impute and normal_values is not None:
            for varid in normal_values.index:
                if df[varid].isnull().all():
                    df[varid] = normal_values[varid]
                else:
                    assert(df[varid].notnull().all())

        df.reset_index(inplace=True)
        return df.as_matrix()

    def age_in_months(self):
        return int(np.floor((self.first_dt - self.dob).total_seconds() / (60 * 60 * 24 * 365) * 12))

    def los_hours(self):
        return np.floor((self.discharge_dt - self.first_dt).total_seconds() / (60 * 60))

    def medical_los_hours(self):
        if self.med_disch_dt is not None:
            return np.floor((self.med_disch_dt - self.first_dt).total_seconds() / (60 * 60))
        else:
            return None

    def delta_pcpc(self):
        return self.discharge_pcpc - self.admit_pcpc

    def delta_popc(self):
        return self.discharge_popc - self.admit_popc

class OldVpVpicuEpisode(VpicuEpisode):

    @staticmethod
    def from_directory(path, varids=[]):
        episodeid = int(re.match('.*/(\d+)', path).group(1))
        sf = pd.read_csv(os.path.join(path, 'static.csv'), index_col=None, parse_dates=[2, 6, 7, 9], na_values=['null']).T.squeeze()
        sex = np.nan if type(sf['SEX'])!=str else (1 if sf['SEX']=='f' else (0 if sf['SEX']=='m' else np.nan))
        weight = float(sf['WEIGHT'])
        dob = sf['DOB'] if type(sf['DOB']) == datetime else sf['DOB'].to_datetime()
        admit = sf['ADMIT'] if type(sf['ADMIT']) == datetime else sf['ADMIT'].to_datetime()
        discharge = sf['DISCHARGE'] if type(sf['DISCHARGE']) == datetime else sf['DISCHARGE'].to_datetime()
        try:
            mdischarge = sf['MEDICALDISCHARGE'] if type(sf['MEDICALDISCHARGE']) == datetime else sf['MEDICALDISCHARGE'].todatetime()
        except:
            mdischarge = None
        died = None if np.isnan(sf['DIED']) else (int(sf['DIED']) > 0)

        cf = pd.read_csv(os.path.join(path, 'codes.csv'), index_col=None, na_values=['null']).T.squeeze()
        apcpc = float(cf['ADMITPCPC'])
        apopc = float(cf['ADMITPOPC'])
        dpcpc = float(cf['DISCHARGEPCPC'])
        dpopc = float(cf['DISCHARGEPOPC'])

        df = pd.read_csv(os.path.join(path, 'diagnoses.csv'), index_col=None, na_values=['null'])
        if df.PRIMARY.sum() > 1:
            raise InvalidVpicuDataException('prim_diag', 'size', df.PRIMARY.sum())
        diags = list(set(df.CODE.tolist()))
        #sdiags = [ '{0:04}'.format(code) for code in set(df.CODE[df.PRIMARY<1].tolist()) ]
        if df.PRIMARY.sum() == 1:
            pdiag = df.CODE[df.PRIMARY>0][0]
        else:
            pdiag = 0

        def convert_value(s):
            s = re.sub('[^\.0123456789]', '', s)
            if s != '':
                return float(s)
            else:
                return np.nan

        pf = pd.read_csv(os.path.join(path, 'variables.csv'), index_col=None, parse_dates=[0], na_values=['null'], converters={ 'VALUE': convert_value })
        pf = pf.sort(columns=['TIME', 'VARID'])
        pf = pf.pivot(index='TIME', columns='VARID', values='VALUE')
        if pf.shape[0] == 0:
            raise InvalidVpicuDataException('msmts', 'size', 0)

        if len(varids)==0:
            varids = range(1,14)
        for c in varids:
            if c not in pf.columns:
                pf[c] = np.nan
        pf[pf==0] = np.nan
        assert((pf==0).sum().sum()==0)

        return VpicuEpisode(episodeid, sex, weight, dob, admit, discharge, died, pdiag, pf, diag=diags, med_disch_dt=mdischarge,
                          admit_pcpc=apcpc, discharge_pcpc=dpcpc, admit_popc=apopc, discharge_popc=dpopc)

