# New Pass
import datetime
import numpy as np
import pandas as pd

from LoadAcoustics import getData



#Parameters
#Cut Right Data
AcousticIndexes = [7,8,9,10,11,12,13,14,15]
TriggerIndex = 15
Trigger_Threshold = 2.0
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



def NewPass(file_list):
    Log = pd.DataFrame(columns = ['file','Start','Stop'])
    
    
    for filename in file_list:
        AcousticData = getData(filename,AcousticIndexes,TriggerIndex,Trigger_Threshold)
        
        temp = {'Start':None,'Stop':None}
        
        if not(type(AcousticData)) == pd.core.frame.DataFrame:   
            temp['file'] = filename
            temp['Start'] = 0#None
            temp['Stop'] = 10 # None
            Log = Log.append(temp, ignore_index=True)
            continue
            
        temp['file'] = filename
        temp['Start'] = AcousticData.index[0]
        temp['Stop'] = AcousticData.index[-1]
        
        Log = Log.append(temp, ignore_index=True)
        
    return Log