from nptdms import TdmsFile
from FirstPassWolf import FirstPass
from SecondPass import SecondPass
from PlottingUtility0 import PlotAll
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt 
from pylab import rcParams
rcParams['figure.figsize'] = 16,12
import os

#%% Load Acoustics Wolf Edition

def getDataFrame(filename):
    if type(filename) == str:
        tdms_file = TdmsFile(filename)#,memmap_dir=cwd)
        tddf = tdms_file.as_dataframe()
    else:
        raise TypeError('I need a single filename')
    return tddf

def getVoltageCare(VoltageIndex,tddf):
    columns = tddf.columns
    voltage_threshold = -0.400
    #IL = pd.rolling_mean(tddf[columns[VoltageIndex]],5000)
    IL = tddf[columns[VoltageIndex]]
    care = np.nonzero(IL < voltage_threshold)[0][0]
    return care

def MakePreciseEnvelope(AcousticData):
    newENV = []
    L = len(AcousticData)
    for i in np.arange(L):
        newENV.append(AIC(AcousticData,i,L))
    return newENV

def AIC(x,k,N):
    return k*np.log(np.var(x[1:k]))+(N - k - 1)*np.log(np.var(x[k+1:N]))

def getStart(envelope):
    DFENV = pd.DataFrame(envelope)
    minima = DFENV.dropna().idxmin().values[0]
    return minima

#%%
def MakeEnvelope(acoustics,factor = 1000): #Balance this 
    acoustics.index = acoustics.index.values//factor
    acoustics = acoustics.abs()
    acoustics = acoustics.groupby(acoustics.index).mean()
    acoustics = acoustics - acoustics.iloc[0]
    c = acoustics.columns
    best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22
    print('envelope created')
    return [acoustics, best_envelope]

def MakeEnvelopeBetter(AcousticData,factor = 1000): #Balance this 
    OffsetFix = AcousticData - AcousticData.iloc[:80000].mean()
    Balanced = OffsetFix / OffsetFix.abs().std()
    
    acoustics = Balanced
    acoustics.index = acoustics.index.values//factor
    acoustics = acoustics.abs()
    acoustics = acoustics.groupby(acoustics.index).mean()
    acoustics = acoustics-acoustics.iloc[:50].mean()
    c = acoustics.columns
    
    #best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22
    best_envelope = (acoustics[c[0]].values*acoustics[c[1]].values)*22
    
    print('envelope created')
    return [acoustics,best_envelope,Balanced]


#AcousticData.iloc[1500000:2000000].plot(alpha=0.6)
#MakeEnvelope(AcousticData.iloc[:500000])[1].plot(alpha=0.6)
#x = MakeEnvelopeBetter(AcousticData,factor=1000000)[2].plot(alpha=0.6)


#%% FirstPass Wolf Edition


#%%
AcousticIndexes = [1,2]
CurrentIndex = 0 #IMAG
VoltageIndex = 3 #IL

tdms_files = glob.glob('*.tdms')
cwd = os.getcwd()

TDDF = getDataFrame(tdms_files[0])
COLS = TDDF.columns
CARE = getVoltageCare(VoltageIndex, TDDF)
AcousticData = TDDF[COLS[[AcousticIndexes]]].iloc[:CARE]

kick_up = 0.00020
pre_kick_level = 0.0015
stop_level = 0.00020 
buffer_pre = 0 #1000 #SET PRE BUFFER FOR MAXIM
buffer_post = 0 #1000
MacroEnvelopeFactor = 1000

PreciseEnvelopeFactor = 200
PreciseCutOff = 10000

MacroEventLog,env_test = FirstPass(AcousticData, factor = MacroEnvelopeFactor, d1_thresh = kick_up, d2_thresh = pre_kick_level, stop_thresh = stop_level, pre_buffer = buffer_pre, post_buffer = buffer_post, filename=tdms_files[0][:-5])


#%%
#
#MacroEventLogSample = MacroEventLog.sample(5)
#PreciseEventLogSample,EnvLog = SecondPass(MacroEventLogSample,AcousticData,tdms_files[0][:-5])
#print(PreciseEventLogSample)
#PlotAll(AcousticData,PreciseEventLogSample,EnvLog,PreciseCutOff,env_test)




PreciseEventLog,EnvLog = SecondPass(MacroEventLog,AcousticData,tdms_files[0][:-5])
#MaxCutLog = PreciseEventLog[['Start','Stop','S_bot time','S_top time']]
#MaxCutLog.to_csv('Events_'+tdms_files[0][:-5]+'.csv')





#%%
#C = PreciseEventLog.columns
#Deltas = PreciseEventLog[C[-1]]
#timebins = np.append(np.arange(0,len(Deltas),235),len(Deltas)-1)
#
#ax = plt.axes()#Deltas.hist(alpha=1)
#for i in np.arange(len(timebins)-1):
#    timewise = Deltas.iloc[timebins[i]:timebins[i+1]]
#    timewise.hist( color= (i/(len(timebins)),0,0), alpha=0.3,bins=200)
#    plt.xlim(-2000,2000)
#    plt.show()
    





