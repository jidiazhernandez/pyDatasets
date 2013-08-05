import os
import re

from pandas import DataFrame


class WaferRun:

    def __init__(self, run_id, wafer_id, label, measurements):
        self.run_id = int(run_id)
        self.wafer_id = int(wafer_id)
        self.label = int(label)
        self.measurements = DataFrame(measurements)
    
    @staticmethod
    def from_files(path, run_id, wafer_id):
        fn_base = os.path.join(path, '{0}_{1:02}'.format(run_id, wafer_id))
        
        try:
            df = DataFrame({11: DataFrame.from_csv(fn_base + '.11', header=None, sep='\t', index_col=None, parse_dates=False)[1],
                            12: DataFrame.from_csv(fn_base + '.12', header=None, sep='\t', index_col=None, parse_dates=False)[1],
                            15: DataFrame.from_csv(fn_base + '.15', header=None, sep='\t', index_col=None, parse_dates=False)[1],
                            6: DataFrame.from_csv(fn_base + '.6', header=None, sep='\t', index_col=None, parse_dates=False)[1],
                            7: DataFrame.from_csv(fn_base + '.7', header=None, sep='\t', index_col=None, parse_dates=False)[1],
                            8: DataFrame.from_csv(fn_base + '.8', header=None, sep='\t', index_col=None, parse_dates=False)[1]})
        except:
            return None
        
        m = re.search('/(normal|abnormal)', path)
        if m is None:
            return None
    
        label = 1 if m.group(1) == 'abnormal' else -1
        
        return WaferRun(run_id, wafer_id, label, df)
    
    def as_nparray(self):
        """Spits out data as a D x T numpy.array (D=# channels, T=# samples)

        Notes:
        Notice what we do here: we start with a pandas.DataFrame where each channel
        is a column (so you can think of it as a T x D matrix). We first rename the
        columns to channel numbers,then sort the columns, then sort the index, then
        transform to numpy.array, then finally take the transpose to get D x T.
        rename(columns=self.wafer_id)
        """
        return self.measurements.sort(axis=1).sort_index().as_matrix().T
