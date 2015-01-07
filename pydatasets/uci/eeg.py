"""This module provides classes and utilities for working with the UCI EEG database located at
http://archive.ics.uci.edu/ml/datasets/EEG+Database. There are 122 subjects total in the dataset,
and for each there is multivariate time series data for about 80-120 trials. In each trial the
subject has a 64 channel scalp EEG placed on his head and one second's worth of data (sampled at
256 Hz, i.e., 256 samples/second) is recorded after the subject is shown one of three visual
stimuli. Each trial, then, is a 64 channel by 256 sample time series.

This module assumes you have downloaded and untarred the eeg_full.tar file, which produces a
folder with the following structure:
eeg_full/
    co2a0000364/
        co2a0000364.rd.000.gz
        co2a0000364.rd.002.gz
        co2a0000364.rd.007.gz
        ...
    ...
    co2c0000337/
        co2c0000337.rd.000.gz
        co2c0000337.rd.002.gz
        co2c0000337.rd.016.gz
        ...
    ...

Each subdirectory of eeg_full is a subject, and each gzipped text file is a trial. The second
character in the directory name ("a" or "c") indicates whether the subject is a member of the
alcoholic or control group. Three digits at the end of the trial file name (000-120) indicate
the trial number.

Each trial file has the following structure:

# [filename minus trial number and extension]
# [experimental info (IGNORED)]
# [sampling rate (IGNORED)]
# [stimulus type] , [trial #]
# [channel name] chan [channel number]
[channel number] [channel name] [timestamp] measurement
[channel number] [channel name] [timestamp] measurement
[channel number] [channel name] [timestamp] measurement
...
# [channel name] chan [channel number]
[channel number] [channel name] [timestamp] measurement
[channel number] [channel name] [timestamp] measurement
[channel number] [channel name] [timestamp] measurement
...

An example being:

# co2a0000364.rd
# 120 trials, 64 chans, 416 samples 368 post_stim samples
# 3.906000 msecs uV
# S1 obj , trial 0
# FP1 chan 0
0 FP1 0 -8.921
0 FP1 1 -8.433
0 FP1 2 -2.574
...
# FP2 chan 1
0 FP2 0 0.834
0 FP2 1 3.276
0 FP2 2 5.717
...
"""

import csv    # this is a plain module import
import gzip
import os
import pickle
import re
import sys

from pandas import DataFrame, Series # how to import something from another module into current namespace

class EegSubject:
    """Reads, parses, prepares, and stores data for one subject.

    Keyword arguments:
    subject_id  -- subject id (e.g., 'co2a0000364')
    alcoholic   -- boolean, whether subject is alcoholic
    trials_data -- list of trials data (could be list of EegTrial objects)
    """

    def __init__(self, subject_id, alcoholic, trials_data=[]):
        self.subject_id = str(subject_id)
        self.alcoholic = bool(alcoholic)
        self.trials_data = trials_data

    @staticmethod
    def from_directory(path, verbosity=1):
        """Static factory method for reading an EegSubject object from a subdirectory of "eeg_full."

        Keyword arguments:
        path -- path to a UCI EEG database subject folder
        verbosity -- level of verbosity
        """

        if verbosity > 0:
            sys.stdout.write('Reading EegSubject from directory ' + path)
            if verbosity > 1:
                sys.stdout.write('\n')
            else:
                sys.stdout.write(':')
        m = re.match('.*/?(co\d(a|c)\d\d\d\d\d\d\d)', path)
        subject_id = m.group(1)
        alcoholic = (m.group(2)=='a')
        filenames = os.listdir(path)
        trials_data = []
        for filename in filenames:
            if verbosity > 1:
                sys.stdout.write('  reading EegTrial from ' + filename + '...')
            else:
                sys.stdout.write('.')
            try:
                trials_data.append(EegTrial.from_file(os.path.join(path, filename)))
            except Exception as e:
                sys.stdout.write('ERROR: could not read EegTrial from file: ' + str(e) + '\n')
            else:
                if verbosity > 1:
                    sys.stdout.write('SUCCESS\n')
        sys.stdout.write('\n')

        return EegSubject(subject_id=subject_id, alcoholic=alcoholic, trials_data=trials_data)

    def to_pickle(self, path='', filename=None):
        """Pickle (serialize) EegSubject object to disk.

        Keyword arguments:
        path -- path without filename to store pickled object
        filename -- filename to append to path, optional
        """

        if filename is None:
            filename = '{0.subject_id}.pkl'.format(self)
        filename = os.path.join(path, filename)
        pickle.dump(self, open(filename, 'w'), pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_pickle(filename):
        """Read EegSubject object from pickle file."""

        return pickle.load(open(filename, 'r'))

    def to_csvs(self, path=''):
        """Store EegSubject to disk as CSVs, one trial per CSV, in a directory.

        Keyword arguments:
        path -- directory to store CSVs
        """

        subdir = os.path.join(path, self.subject_id)
        try:
            os.makedirs(subdir)
        except:
            pass
        for trial in self.trials_data:
            trial.to_csv(path=subdir)



class EegTrial:
    """Reads, parses, prepares, and stores data for one trial.

    Keyword arguments:
    subject_id  -- subject id (e.g., 'co2a0000364')
    alcoholic   -- boolean, whether subject is alcoholic
    trial_num   -- trial number, 000-120
    stimulus    -- type of stimulus (raw string is fine)
    data        -- measurements data
    trials_data -- dictionary mapping channel name to channel number

    Notes:
    Will attempt to coerce "data" into a pandas.DataFrame so you should pass
    in something that is compatible (just pass a DataFrame!). Probably "channels"
    should be a plain dictionary with channel name as key, channel number as value.
    This is overkill: the channels appeared to be numbered consistently throughout
    the UCI EEG data set, but it's good practice: they don't HAVE to be numbered
    consistently...
    """

    def __init__(self, subject_id, alcoholic, trial_num, stimulus, data, channels):
        self.subject_id = str(subject_id)
        self.alcoholic = bool(alcoholic)
        self.trial_num = int(trial_num)
        self.stimulus = str(stimulus)
        self.data = DataFrame(data)
        self.channels = channels
        self.data.sort_index(inplace=True)
        self.data.sort(axis=1, inplace=True)
     
    @staticmethod
    def from_file(filename):
        """Static factory method for reading an EegTrial object from a (gzipped?) file in "eeg_full."

        Keyword arguments:
        filename -- path plus filename to a UCI EEG database file
        """

        if filename.endswith('.gz'):
            f = gzip.open(filename, 'r')
        else:
            f = open(filename, 'r')
        l = f.readline()
        m = re.match('# (co\d(a|c)\d\d\d\d\d\d\d)', l) # line 1 has subject id
        subject_id = m.group(1)
        alcoholic = (m.group(2)=='a')
        f.readline() # skip line 2
        f.readline() # skip line 3
        l = f.readline()
        m = re.match('# (.*?), trial (\d+)', l) # line 4 has stimulus, trial number
        stimulus = re.sub('\W', '', m.group(1))
        trial_num = int(m.group(2))
        # begin real data, one channel at a time
        data = dict()
        o = []
        t = []
        channels = dict()
        curr = None
        for row in csv.reader(f, delimiter=' '):
            if row[0] == '#':
                if curr is not None:
                    data[curr] = Series(data=o, index=t)
                o = []
                t = []
                assert(row[1] not in channels)
                channels[row[1]] = int(row[3])
                curr = row[1]
            else:
                assert(curr is None or curr == row[1])
                t.append(int(row[2]))
                o.append(float(row[3]))
        if curr is not None:
            data[curr] = Series(data=o, index=t)
        f.close()

        return EegTrial(subject_id=subject_id, alcoholic=alcoholic, trial_num=trial_num, stimulus=stimulus, data=DataFrame(data), channels=Series(channels))

    def get_channel_names(self):
        return list(self.data.columns)
    
    def get_channel_numbers(self):
        return [ self.channels[c] for c in self.data.columns ]

    def as_nparray(self):
        """Spits out data as a D x T numpy.array (D=# channels, T=# samples)

        Notes:
        Notice what we do here: we start with a pandas.DataFrame where each channel
        is a column (so you can think of it as a T x D matrix). We first rename the
        columns to channel numbers,then sort the columns, then sort the index, then
        transform to numpy.array, then finally take the transpose to get T x D.
        """
        return self.data.as_matrix()

    def to_pickle(self, path='', filename=None):
        """Pickle (serialize) EegTrial object to disk.

        Keyword arguments:
        path -- path without filename to store pickled object
        filename -- filename to append to path, optional
        """

        if filename is None:
            filename = '{0.subject_id}-{1}-{0.trial_num:03}-{0.stimulus}.pkl'.format(self, 'a' if self.alcoholic else 'c')
        filename = os.path.join(path, filename)
        pickle.dump(self, open(filename, 'w'), pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_pickle(filename):
        """Read EegTrial object from pickle file."""

        return pickle.load(open(filename, 'r'))

    def to_csv(self, path='', filename_base=None):
        """Store EegTrial to disk as three CSVs: data, channel dictionary, info.

        Keyword arguments:
        path -- directory to store CSVs
        filename_base -- base of filename (extensions will be appended)
        """

        if filename_base is None:
            filename_base = '{0.subject_id}-{1}-{0.trial_num:03}-{0.stimulus}'.format(self, 'a' if self.alcoholic else 'c')
        filename_base = os.path.join(path, filename_base)
        temp = self.data.rename(columns=self.channels).sort(axis=1).sort_index()
        temp.to_csv(filename_base + '.data', header=False, index=False)
        self.channels.to_csv(filename_base + '.channels')
        f = open(filename_base + '.info', 'w')
        f.write('subject_id,{0.subject_id}\nalcoholic,{0.alcoholic}\ntrial_num,{0.trial_num}\nstimulus,{0.stimulus}'.format(self))
        f.close()
