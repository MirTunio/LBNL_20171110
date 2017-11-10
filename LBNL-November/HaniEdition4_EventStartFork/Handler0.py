'''
Handler: Goes through big pile of tdms files and schedules processing steps

imports all the  other steps and runs for each ramping step

'''
#%load_ext autoreload
#%autoreload 2

#Standard modules
import numpy as np
import pandas as pd
import glob

#Processing modules
from LoadAcoustics import getData
from LoadAcoustics import getDataFrame
from LoadAcoustics import getAcousticData

from FirstPass import FirstPass
from SecondPass import SecondPass
from PlottingUtility0 import PlotAll

from matplotlib import pyplot as plt 

#Parameters
#Cut Right Data
AcousticIndexes = [7,8,9,10,11,12,13,14,15]
TriggerIndex = 15
Trigger_Threshold = 1.0
MacroEnvelopeFactor = 100

#Rise and Stop Triggering, Buffers
'''WORKING:
kick_up = 0.00020, aggressive 0.00070, atnoise = 0.00030
pre_kick_level = 0.0015
stop_level = 0.00020 
buffer_pre = 1000
buffer_post = 1000'''

kick_up = 3#0.00030
pre_kick_level = 3#0.0015
stop_level = 3#0.00030 
buffer_pre = 0 #1000 #SET PRE BUFFER FOR MAXIM
buffer_post = 0 #1000

PreciseEnvelopeFactor = 20
PreciseCutOff = 10000
#%% LOAD ALL
#filename = tdms_files[0]
tdms_files = glob.glob('*.tdms')

##%%3
#from PlottingUtility0 import PlotAll
#from SecondPass import SecondPass
#
#
#for i in tdms_files:
#    filename = i
#    print(i)
#    AcousticData = getData(filename,AcousticIndexes,TriggerIndex,Trigger_Threshold)#,manual=True)
#    MacroEventLog,env_test = FirstPass(AcousticData, factor = MacroEnvelopeFactor, d1_thresh = kick_up, d2_thresh = pre_kick_level, stop_thresh = stop_level, pre_buffer = buffer_pre, post_buffer = buffer_post, filename=filename[:-5])
#    PreciseEventLogSample,EnvLog = SecondPass(MacroEventLog,AcousticData,filename[:-5])
#    PlotAll(AcousticData,PreciseEventLogSample,EnvLog,PreciseCutOff,env_test)
#    plt.show()


from NewPass import NewPass

NewLog = NewPass(tdms_files).dropna()

for i in np.arange(len(NewLog)):
    event = NewLog.iloc[i]
    filename = event.file
    START = event.Start
    STOP = event.Stop
    
    df = getDataFrame(filename)
    acousticData = getAcousticData(AcousticIndexes,df.iloc[START:STOP])
    
    acousticData.plot()
    print(filename)
    plt.show()