'''
Acoustic loader: Loads tdms files to memory, cuts out pre quench data and yields acoustic channels

'''
from nptdms import TdmsFile
import numpy as np
import pandas as pd

def getDataFrame(filename):
    if type(filename) == str:
        tdms_file = TdmsFile(filename)
        tddf = tdms_file.as_dataframe()
    else:
        raise TypeError('I need a single filename')
    return tddf


def getAcousticData(AcousticIndexes, tddf):
    if type(AcousticIndexes) == list:
        columns = tddf.columns
        concernedWith = columns[AcousticIndexes]
    else:
        raise TypeError('I need a list of column indexes where acoustic data is located')
				
    return tddf[concernedWith]


def cutQuench(tddf, trigger_index, trigger_threshold, manual = False, manual_index = 110000000):
    if manual:
        return tddf.head(manual_index)
    else:
        c = tddf.columns 
        trigger_data = np.abs(tddf[c[trigger_index]])
        
        try:
            care = np.nonzero(trigger_data > trigger_threshold)[0][0]
        except:
            print('ERROR')
            return False
            
        if care < 2500:
            print('ERROR')
            return False
        
        cut_tddf = tddf.iloc[care-60000:care+15000]#[care+1500:care+5000]
    
    print('removed post quench data')
    return cut_tddf
    

def getData(filename,AcousticIndexes,trigger_index,trigger_threshold, manual = False, manual_index = 110000000):
    df = getDataFrame(filename)
    cut_df = cutQuench(df,trigger_index,trigger_threshold, manual = manual, manual_index = manual_index)
    
    if not(type(cut_df) == pd.core.frame.DataFrame):        
        return False
        
    acousticData = getAcousticData(AcousticIndexes,cut_df)
    
    print(filename + ' loaded successfully')
    return acousticData