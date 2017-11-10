#Opens several short 8 channel tdms files to log quench start times

from nptdms import TdmsFile
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt 
from pylab import rcParams
rcParams['figure.figsize'] = 16,12

AcousticIndexes = [8,9,10,11,12,13,14,15]
VoltageIndexes = [2,3]
TriggerIndex = 0


tdms_files = glob.glob('*.tdms')

def getDataFrame(filename):
    if type(filename) == str:
        tdms_file = TdmsFile(filename)
        tddf = tdms_file.as_dataframe()
    else:
        raise TypeError('I need a single filename')
    return tddf

def getTriggerCare(AcousticIndexes, Trigger_Index, tddf):
    columns = tddf.columns
    trigger_threshold = 3
    trigger_data = tddf[columns[Trigger_Index]]
    
    try:
        care = np.nonzero(trigger_data < trigger_threshold)[0][0]
    except:
        return False
    
    return care 

def getVoltageCare(VoltageIndexes,tddf):
    columns = tddf.columns
    crossover_threshold = 0.06
    
    IL = pd.rolling_mean(tddf[columns[VoltageIndexes[0]]],500)
    OL = pd.rolling_mean(tddf[columns[VoltageIndexes[1]]],500)
    
    IL = IL-IL.dropna().iloc[:200].mean()
    OL = OL-OL.dropna().iloc[:200].mean()
    
    VDIF = np.abs(IL-OL)
    care = np.nonzero(VDIF > crossover_threshold)[0][0]
    
    return care

def MakePreciseEnvelope(AcousticData):
    raw = AcousticData
    newENV = []
    
    for i in np.arange(len(raw)):
        newENV.append(AIC(raw,i,len(raw)))

    return newENV

def AIC(x,k,N):
    return k*np.log(np.var(x[1:k]))+(N - k - 1)*np.log(np.var(x[k+1:N]))

#%% Cuts window around quench for each tdms file
  
LOG = pd.DataFrame(columns = ['File','Start','Stop'])

for file in tdms_files:

    temp = {'File':file,'Start':None,'Stop':None}
    TDDF = getDataFrame(file)
    CARE = getVoltageCare(VoltageIndexes, TDDF)
    
    if CARE:
        temp['Start'] = CARE - 4000
        temp['Stop'] = CARE
        
        print(file + ' LOADED')
    else:
        print('AN ERROR OCCURED LOADING: ' + file)
        continue
    
    LOG = LOG.append(temp, ignore_index=True)

def getStart(envelope):
    DFENV = pd.DataFrame(envelope)
    minima = DFENV.dropna().idxmin().values[0] 

    return minima#np.argmin(envelope)

##############################################################################3
#%% Finds accurate start times for each channel in each file

BESTLOG = pd.DataFrame(columns = ['Quench No.','File','Start','Stop','S1','S2','S3','S4','S5','S6','S8','S9'])


for i in np.arange(len(LOG)):
    QUENCH = LOG.iloc[i]
    FILE = QUENCH.File
    START = QUENCH.Start
    STOP = QUENCH.Stop
    
    print('Processing File: ' + FILE)
    
    temp = {'Quench No.':None,'File':FILE[:-5],'Start':START,'Stop':STOP,'S1':None,'S2':None,'S3':None,'S4':None,'S5':None,'S6':None,'S8':None,'S9':None}
    
    TDDF = getDataFrame(FILE).iloc[START:STOP]
    COLS = TDDF.columns
    
    for colnum in AcousticIndexes:
        channel_data = TDDF[COLS[colnum]]
        channel_dex = channel_data.index
        channel_val = channel_data.values
        trigger = TDDF[COLS[15]].values
        
        channel_env = MakePreciseEnvelope(channel_val)
        
        numdex = FILE.index('Q')
        goodname0 = FILE[numdex+1:]
        dashdex = goodname0.index('_')
        goodname1 = goodname0[:dashdex]
        temp['Quench No.'] = int(goodname1)
        
        if colnum == 8:
            temp['S1'] = getStart(channel_env)
        elif colnum == 10:
            temp['S3'] = getStart(channel_env)
        elif colnum == 12:
            temp['S5'] = getStart(channel_env)
        elif colnum == 14:
            temp['S8'] = getStart(channel_env)
        elif colnum == 15:
            temp['S9'] = getStart(channel_env)
        elif colnum == 9:
            temp['S2'] = getStart(channel_env)
        elif colnum == 11:
            temp['S4'] = getStart(channel_env)
        elif colnum == 13:
            temp['S6'] = getStart(channel_env)            

    BESTLOG = BESTLOG.append(temp,ignore_index=True)
BESTLOG.index = BESTLOG['Quench No.']
BESTLOG = BESTLOG.sort_values('Quench No.')

del BESTLOG['Quench No.']
BESTLOG.to_csv('BESTLOG_4.csv')

#%% Runs a second pass on all old start times
UBERLOG = pd.DataFrame(columns = ['Quench No.','File','Start','Stop','S1','S2','S3','S4','S5','S6','S8','S9'])


for i in np.arange(len(BESTLOG)):
    BESTDATA = BESTLOG.iloc[i]
    FILE = BESTDATA.File+'.tdms'
    START = BESTDATA.Start
    STOP = BESTDATA.Stop
    
    print('Uber Processing File: ' + FILE)
    
    temp = {'Quench No.':None,'File':FILE[:-5],'Start':START,'Stop':STOP,'S1':None,'S2':None,'S3':None,'S4':None,'S5':None,'S6':None,'S8':None,'S9':None}
    
    
    COLS = TDDF.columns
    
    AFTBUFFER = 1000
    FOREBUFFER = 500
    
    for colnum in AcousticIndexes:        
        #Getting quench number from filename
        numdex = FILE.index('Q')
        goodname0 = FILE[numdex+1:]
        dashdex = goodname0.index('_')
        goodname1 = goodname0[:dashdex]
        temp['Quench No.'] = int(goodname1)
        
        if colnum == 8:
            oldMARK = START + BESTDATA['S1']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)          
            temp['S1'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        elif colnum == 10:
            oldMARK = START + BESTDATA['S3']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)
            temp['S3'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        elif colnum == 12:
            oldMARK = START + BESTDATA['S5']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)
            temp['S5'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        elif colnum == 14:
            oldMARK = START + BESTDATA['S8']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)
            temp['S8'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        elif colnum == 15:
            oldMARK = START + BESTDATA['S9']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)
            temp['S9'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        elif colnum == 9:
            oldMARK = START + BESTDATA['S2']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)
            temp['S2'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        elif colnum == 11:           
            oldMARK = START + BESTDATA['S4']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)
            temp['S4'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        elif colnum == 13:
            oldMARK = START + BESTDATA['S6']
            TDDF = getDataFrame(FILE).iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]
            channel_data = TDDF[COLS[colnum]]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)
            temp['S6'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START    

    UBERLOG = UBERLOG.append(temp,ignore_index=True)
UBERLOG.index = UBERLOG['Quench No.']
UEBRLOG = UBERLOG.sort_values('Quench No.')

del UBERLOG['Quench No.']
UBERLOG.to_csv('UBERLOG_4_should_work.csv')

BESTLOG = UBERLOG.copy()