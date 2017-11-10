'''
Second Pass

'''

import pandas as pd
import numpy as np
import datetime

def MakePreciseEnvelope(AcousticData,precise_factor = 200,precise_cutoff=5000):
    dex = AcousticData.iloc[:precise_cutoff].index
    c = AcousticData.columns
    
    valA = AcousticData[c[0]].values[:precise_cutoff]
    valB = AcousticData[c[1]].values[:precise_cutoff]
    
    newA = []
    newB = []
    
    for i in np.arange(len(valA)):
        newA.append(AIC(valA,i,len(valA)))
        newB.append(AIC(valB,i,len(valA)))

    newEnvelope = pd.DataFrame()
    newEnvelope[c[0]] = newA
    newEnvelope[c[1]] = newB
    
    newEnvelope = newEnvelope/100000
    newEnvelope = newEnvelope - newEnvelope.iloc[30]
    
    newEnvelope.index = dex
    return newEnvelope

def AIC(x,k,N):
    return k*np.log(np.var(x[1:k]))+(N - k - 1)*np.log(np.var(x[k+1:N]))


def get_precise(env):
    c = env.columns
    ok = env.replace([np.inf,-np.inf],np.nan).dropna().idxmin()
    precise_startA, precise_startB = ok[c[0]],ok[c[1]]
#    from scipy import signal
#    widths = np.arange(500, 1000)
#    clean = env.replace([np.inf,-np.inf],np.nan).dropna()
#    A = -1*clean[c[0]].values
#    B = -1*clean[c[1]].values
#    
#    peaksA = signal.find_peaks_cwt(A, widths)
#    peaksB = signal.find_peaks_cwt(B, widths)
#
#    precise_startA, precise_startB = peaksA[0],peaksB[0]
#
#    print(precise_startA, precise_startB)
#    
    return precise_startA, precise_startB
    #return precise_startA+env.index[0], precise_startB+env.index[0]


def SecondPass(MacroEventLog,AcousticData,filename):
    EntropyEnvLog = pd.DataFrame()
    PreciseLog = pd.DataFrame(columns = ['S_bot time', 'S_top time', 'S_top - S_bot'])
    dex = MacroEventLog.index
    
    for i in np.arange(len(MacroEventLog)):
        vent = MacroEventLog.iloc[i]
#        strt = vent.Start
#        stop = vent.Stop
        
        strt = MacroEventLog.iloc[i].Story[0]
        stop = MacroEventLog.iloc[i].Story[1]
        
        EntropyEnvelope = MakePreciseEnvelope(AcousticData[strt:stop]) 
        
        c = EntropyEnvelope.columns
        EntropyEnvTemp = {c[0]:EntropyEnvelope[c[0]],c[1]:EntropyEnvelope[c[1]]}
        EntropyEnvLog = EntropyEnvLog.append(EntropyEnvTemp, ignore_index=True)
        
        precise_A, precise_B = get_precise(EntropyEnvelope)
        temp = {'S_bot time':precise_A, 'S_top time':precise_B, 'S_top - S_bot':precise_B-precise_A}
        
        PreciseLog = PreciseLog.append(temp, ignore_index=True)
        
    gentime = str(datetime.datetime.now())[:-7].replace(':','_').replace(' ','  ').replace('-','_')    
    PreciseLog.index = dex
    PreciseEventLog = pd.concat([MacroEventLog,PreciseLog],axis=1)#.sort_index()  
    #del PreciseEventLog[PreciseEventLog.columns[0]]
    
    PreciseEventLog.to_csv(filename + ' ' + gentime + '_PRECISE.csv')
    
    print('Precise log generated')
    return PreciseEventLog,EntropyEnvLog
    













