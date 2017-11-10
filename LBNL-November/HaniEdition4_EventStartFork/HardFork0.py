# Quench Timer 0

from nptdms import TdmsFile
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt 

AcousticIndexes = [7,9,11,13,14]#[7,8,9,10,11,12,13,14,15]
TriggerIndex = 15


tdms_files = glob.glob('*.tdms')
#tdms_files = ['8_16_2017_3_22_39_PM.tdms']

def getDataFrame(filename):
    if type(filename) == str:
        tdms_file = TdmsFile(filename)
        tddf = tdms_file.as_dataframe()
    else:
        raise TypeError('I need a single filename')
    return tddf

def getCare(AcousticIndexes, Trigger_Index, tddf):
    columns = tddf.columns
    trigger_threshold = 3
    trigger_data = tddf[columns[Trigger_Index]]
    
    try:
        care = np.nonzero(trigger_data < trigger_threshold)[0][0]
    except:
        return False
    
    return care 
    
def MakePreciseEnvelope(AcousticData):
    raw = AcousticData
    newENV = []
    
    for i in np.arange(len(raw)):
        newENV.append(AIC(raw,i,len(raw)))
        
    return newENV

def AIC(x,k,N):
    return k*np.log(np.var(x[1:k]))+(N - k - 1)*np.log(np.var(x[k+1:N]))

#%%
  
LOG = pd.DataFrame(columns = ['File','Start','Stop'])

for file in tdms_files:
    temp = {'File':file,'Start':None,'Stop':None}
    TDDF = getDataFrame(file)
    CARE = getCare(AcousticIndexes, TriggerIndex, TDDF)
  
    if CARE:
        temp['Start'] = CARE - 30000
        temp['Stop'] = CARE# + 10000
        
        print(file + ' LOADED')
    else:
        print('AN ERROR OCCURED LOADING: ' + file)
        continue
    
    LOG = LOG.append(temp, ignore_index=True)

def getStart(envelope):
    DFENV = pd.DataFrame(envelope)
    minima = DFENV.dropna().idxmin().values[0]
    return minima#np.argmin(envelope)

#%%

BESTLOG = pd.DataFrame(columns = ['File','Start','Stop','S1','S3','S5'])


#for i in np.arange(len(LOG)):
#    QUENCH = LOG.iloc[i]
#    FILE = QUENCH.File
#    START = QUENCH.Start
#    STOP = QUENCH.Stop
#    
#    print('Processing File: ' + FILE)
#    
#    temp = {'File':FILE,'Start':START,'Stop':STOP,'S1':None,'S3':None,'S5':None,'S8':None,'S9':None}
#    
#    TDDF = getDataFrame(FILE).iloc[START:STOP]#.iloc[START-10000:STOP+10000]
#    COLS = TDDF.columns
#    
#    for colnum in AcousticIndexes:
#        channel_data = TDDF[COLS[colnum]]
#        channel_dex = channel_data.index
#        channel_val = channel_data.values
#        trigger = TDDF[COLS[15]].values
#        
#        channel_env = MakePreciseEnvelope(channel_val)
#        
#        if colnum == 7:
#            temp['S1'] = getStart(channel_env)
#        elif colnum == 9:
#            temp['S3'] = getStart(channel_env)
#        elif colnum == 11:
#            temp['S5'] = getStart(channel_env)
#        elif colnum == 13:
#            temp['S8'] = getStart(channel_env)
#        elif colnum == 14:
#            temp['S9'] = getStart(channel_env)
##        plt.plot(channel_val)
##        plt.plot(trigger/10)
##        #plt.show()
##
##        plt.plot(np.array(channel_env)/420000)
##        plt.plot(trigger/10)
##        plt.show()
##        
##        print(COLS[colnum],FILE)
#    BESTLOG = BESTLOG.append(temp,ignore_index=True)

'''
> Consider using current to make starts happen, cut around that
> Make sure starts are accurate

'''

def inspect(BESTLOG):
    for i in np.arange(len(BESTLOG)):
        pass
        
        
        
        
        