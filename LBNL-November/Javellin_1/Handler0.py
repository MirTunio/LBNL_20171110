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
from FirstPass import FirstPass
from SecondPass import SecondPass
from PlottingUtility0 import PlotAll

#Parameters
#Cut Right Data
AcousticIndexes = [1,2]
TriggerIndex = 0
Trigger_Threshold = 0.3
MacroEnvelopeFactor = 1000

#Rise and Stop Triggering, Buffers
'''WORKING:
kick_up = 0.00020
pre_kick_level = 0.0015
stop_level = 0.00020 
buffer_pre = 1000
buffer_post = 1000
'''
kick_up = 0.00020
pre_kick_level = 0.0015
stop_level = 0.00020 
buffer_pre = 0 #1000 #SET PRE BUFFER FOR MAXIM
buffer_post = 0 #1000

PreciseEnvelopeFactor = 200
PreciseCutOff = 10000
#%% LOAD SINGLE
tdms_files = glob.glob('*.tdms')
filename = tdms_files[0]
AcousticData = getData(filename,AcousticIndexes,TriggerIndex,Trigger_Threshold,manual=True)

#%% PROCESS SINGLE
MacroEventLog,env_test = FirstPass(AcousticData, factor = MacroEnvelopeFactor, d1_thresh = kick_up, d2_thresh = pre_kick_level, stop_thresh = stop_level, pre_buffer = buffer_pre, post_buffer = buffer_post, filename=filename[:-5])

#%% EXPLORE

MacroEventLogSample = MacroEventLog.sample(20)
PreciseEventLogSample,EnvLog = SecondPass(MacroEventLogSample,AcousticData,filename[:-5])
print(PreciseEventLogSample)
PlotAll(AcousticData,PreciseEventLogSample,EnvLog,PreciseCutOff,env_test)












