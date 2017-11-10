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
import datetime

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
    
    IL = tddf[columns[VoltageIndex]]
    care = np.nonzero(IL < voltage_threshold)[0][0]
    return care

#%% FirstPass Wolf Edition
def MacroEnvelope(AcousticData,factor = 500):  
    OffsetFix = AcousticData - AcousticData.iloc[:80000].mean()
    Balanced = OffsetFix / OffsetFix.abs().std()
    
    acoustics = Balanced
    acoustics.index = acoustics.index.values//factor
    acoustics = acoustics.abs()
    acoustics = acoustics.groupby(acoustics.index).mean()
    acoustics = acoustics-acoustics.iloc[:50].mean()
    c = acoustics.columns
    
    best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22
    
    print('envelope created')
    return best_envelope



#%%
def RiseMark(env,d1_thresh = 12, d2_thresh = 5): #12
    '''
    riser anatomy: _./
    d1 thresh: The amount it rises after potential riser "  ./  "
    d2 thresh: The quiet it must see before the potential riser " _. "
    '''
    
    dif = env.diff()
    d1 = (dif > d1_thresh)
    d1.index = d1.index + 1
    
    dif = env.diff()
    d2 = (dif < d2_thresh)
    d2.index = d1.index + 1	
    	
    T = d1 & d2	
    T.index = T.index - 2

    c = T.columns
    T = T[c[0]]&T[c[0]]
    
    T2 = T.tail(len(T)-1)
    T2.index = T2.index-1
    T = T2&(T^T2)
    
    T.index = T.index + 2
    return T
    
def StopMark(env, stop_thresh = 15):
    diff = np.abs(env.diff())
    d1 = diff < stop_thresh
    d2 = diff < stop_thresh
    d3 = diff < stop_thresh
    
    d2.index = d2.index+1
    d3.index = d2.index+2
    
    T = (d1 & d2) & d3
    
    #T2 = T.tail(len(T)-1)
    #T2.index = T2.index-1
    #T = T2&(T^T2)
    
    c = T.columns
    T = T[c[0]]|T[c[0]]
    #T.index = T.index + 1
    return T


def EventCutter(rise,stop,factor = 1000, pre_buffer = 1000, post_buffer = 1000, filename='TEST'):
    stop.iloc[-1] = True
    Log = pd.DataFrame(columns = ['Start','Stop','SubEvents','Story'])
    
    State = False
    logic = 2*rise - stop    
    
    temp = {'Start':None,'Stop':None,'SubEvents':None, 'Story':None}
    subno = 1
    story = []
    
    for i in np.arange(len(logic)):
        here = logic.iloc[i]
        
        if State == True and here > 0:
            story.append(i*factor)
            subno +=1
            
        elif State == False and here > 0:
            State = True
            story.append(i*factor-pre_buffer)
            temp['Start'] = i*factor - pre_buffer
        
        elif State == True and here < 0:
            story.append(i*factor + post_buffer)
            temp['Stop'] = i*factor + post_buffer
            temp['SubEvents'] = subno
            temp['Story'] = np.array(story)        
            Log = Log.append(temp, ignore_index=True)
            
            temp = {'Start':None,'Stop':None,'SubEvents':None, 'Story':None}
            subno = 1
            story = []
            State = False
            
    Log.index.name = 'Event No.'
    return Log  

#%%MODULE MODULE Standard load

AcousticIndexes = [1,2]
CurrentIndex = 0 #IMAG
VoltageIndex = 3 #IL

tdms_files = glob.glob('*.tdms')

TDDF = getDataFrame(tdms_files[0])
COLS = TDDF.columns
CARE = getVoltageCare(VoltageIndex, TDDF)
AcousticData = TDDF[COLS[[AcousticIndexes]]].iloc[:CARE]

#%% Testing Risers and Stops again
ENVELOPE = MacroEnvelope(AcousticData)
print('Macro envelope created...')
#%%
RISE = RiseMark(ENVELOPE)
STOP = StopMark(ENVELOPE, stop_thresh = 30)
print('Risers and stops marked...')

#%%TESTING TESTING  Envelope, Rise and Stop testing
ax = ENVELOPE.iloc[-200:].plot(linewidth=2,color='black')
ENVELOPE.iloc[-200:].diff().plot(ax=ax,linewidth=1,color='blue')
logic = 2*RISE-STOP
(logic*40).iloc[-200:].plot(ax = ax)
(RISE.iloc[-200:][RISE!=False]).plot(ax=ax,color='green',linewidth=0,alpha=0.6,marker='o',markerSize=12)
(STOP.iloc[-200:][STOP!=False]).plot(ax=ax,color='red',linewidth=0,alpha=0.6,marker='o',markerSize=12)
plt.ylim(-40,150)
plt.show()
AcousticData.iloc[-100000:].plot(alpha=0.6)
plt.show()

#%%
MacroLog = EventCutter(RISE,STOP)
print(str(len(MacroLog)) + ' Macro Events cut...')

#%% Event cutter testing
mStart = MacroLog.Start
mStops = MacroLog.Stop

CUT = AcousticData.iloc[-200000:]#-100000:]
ax = CUT.plot(alpha=0.6)

for stt in mStart[mStart>CUT.index[0]][mStart<CUT.index[-1]]:
    plt.plot(stt,0,linewidth = 0, marker = 'o', color = 'green', markerSize = 30,alpha=0.6)
    
for stp in mStops[mStops>CUT.index[0]][mStops<CUT.index[-1]]:
    plt.plot(stp,0,linewidth = 0, marker = 'o', color = 'red', markerSize = 20,alpha=0.6)

#ENVELOPE.index = ENVELOPE.index*1000
#(ENVELOPE[(ENVELOPE.index>CUT.index[0]) & (ENVELOPE.index<CUT.index[-1])]/1000).plot(ax=ax,linewidth=2,color='black')
ENV = ENVELOPE.copy()
ENV.index = ENV.index*1000
(ENV[(ENV.index > CUT.index[0]) & (ENV.index < CUT.index[-1])]/1000).diff().abs().plot(ax=ax,linewidth=1,color='blue',marker='x')

#STAP = STOP.copy()
#STAP.index = STAP.index*1000
#(STAP[(STAP.index>CUT.index[0]) & (STAP.index<CUT.index[-1])][STOP!=False]).plot(ax=ax,color='red',linewidth=0,alpha=0.6,marker='o',markerSize=20)

plt.ylim(-0.060,0.060)
plt.show()

print('done')


#%%
#%%  SecondPass Wolf Edition

def MakePreciseEnvelope(AcousticDataIN):
    newENV = []
    L = len(AcousticDataIN)
    for i in np.arange(L):
        newENV.append(AIC(AcousticDataIN,i,L))
    return newENV

def AIC(x,k,N):
    return k*np.log(np.var(x[1:k]))+(N - k - 1)*np.log(np.var(x[k+1:N]))

def getStart(envelope):
    DFENV = pd.DataFrame(envelope)
    minima = DFENV.dropna().idxmin().values[0]
    return minima

def SecondPass(MacroEventLog,AcousticData,filename):
    PreciseLog = pd.DataFrame(columns = ['S_bot', 'S_top', 'S_top - S_bot'])
    dex = MacroEventLog.index
    C = AcousticData.columns
    
    for i in np.arange(len(MacroEventLog)):
        vent = MacroEventLog.iloc[i]
        strt = vent.Start
        stop = vent.Stop
        
        EntropyEnvelopeA = MakePreciseEnvelope(AcousticData[C[0]].iloc[strt+00:strt+2200]) 
        EntropyEnvelopeB = MakePreciseEnvelope(AcousticData[C[1]].iloc[strt+00:strt+2200])
        
        precise_A = getStart(EntropyEnvelopeA) + 00
        precise_B = getStart(EntropyEnvelopeB) + 00
        
        temp = {'S_top':precise_A, 'S_bot':precise_B, 'S_top - S_bot':precise_B-precise_A}
        
        PreciseLog = PreciseLog.append(temp, ignore_index=True)
        
    PreciseLog.index = dex
    PreciseEventLog = pd.concat([MacroEventLog,PreciseLog],axis=1)

    print('Precise log generated')
    return PreciseEventLog

def ThirdPass(PreciseLog, AcousticData, filename):
    UBERLOG = pd.DataFrame(columns = ['File','Start','Stop','S_top','S_bot'])

    for i in np.arange(len(PreciseLog)):
        BESTDATA = PreciseLog.iloc[i]
        START = BESTDATA.Start
        STOP = BESTDATA.Stop
        FILE = filename
        
        temp = {'File':FILE[:-5],'Start':START,'Stop':STOP,'S_top':None,'S_bot':None}
    
        C = TDDF.columns
    
        AFTBUFFER = 1000
        FOREBUFFER = 500
    
        
        oldMARK = START + BESTDATA['S_bot']
        channel_data = AcousticData.iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]["/'3.051758E-5 ; 6.103516E-5 ; 6.103516E-5 ; 1.525879E-5 ; '/'S_bot'"]
        channel_val = channel_data.values  
        channel_env = MakePreciseEnvelope(channel_val)          
        temp['S_bot'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
            
        oldMARK = START + BESTDATA['S_top']
        channel_data = AcousticData.iloc[oldMARK - FOREBUFFER:oldMARK + AFTBUFFER]["/'3.051758E-5 ; 6.103516E-5 ; 6.103516E-5 ; 1.525879E-5 ; '/'S_top'"]
        channel_val = channel_data.values  
        channel_env = MakePreciseEnvelope(channel_val)          
        temp['S_top'] = getStart(channel_env)  + oldMARK - FOREBUFFER - START
               
        UBERLOG = UBERLOG.append(temp,ignore_index=True)
        UBERLOG.index.name = 'Event No.'
    UBERLOG.index = PreciseLog.index
    print('Uber Log generated')
    return UBERLOG

#%% Standard Load
#AcousticIndexes = [1,2]
#CurrentIndex = 0 #IMAG
#VoltageIndex = 3 #IL
#
#tdms_files = glob.glob('*.tdms')
#
#TDDF = getDataFrame(tdms_files[0])
#COLS = TDDF.columns
#CARE = getVoltageCare(VoltageIndex, TDDF)
#AcousticData = TDDF[COLS[[AcousticIndexes]]].iloc[:CARE]

#%%
ENVELOPE = MacroEnvelope(AcousticData)

#%%
RISE = RiseMark(ENVELOPE)
STOP = StopMark(ENVELOPE, stop_thresh = 5)
MacroLog = EventCutter(RISE,STOP,factor=500)

RISE.index = RISE.index*500
STOP.index = STOP.index*500

#%%
PreciseLog = SecondPass(MacroLog.sample(10), AcousticData, tdms_files[0])
UberLog = ThirdPass(PreciseLog, AcousticData, tdms_files[0])

# Testing precise starts

for i in np.arange(len(UberLog)):
    Events = UberLog.index
    Pcut = PreciseLog.iloc[i]
    Ucut = UberLog.iloc[i]
    
    strt = Ucut.Start
    stop = Ucut.Stop
    
    P_top = Pcut.S_top + strt
    P_bot = Pcut.S_bot + strt
    
    U_top = Ucut.S_top + strt
    U_bot = Ucut.S_bot + strt
    
    ax = AcousticData.iloc[strt-1500:stop+1500].plot(alpha=0.6)
    
    #S_top is BLUE, S_bot is ORANGE
    plt.plot(P_top, 0,  marker='o', color = 'blue', markerSize = 12, alpha=0.7)
    plt.plot(U_top, 0,  marker='*', color = 'blue', markerSize = 12, alpha=0.7)
    
    plt.plot(P_bot, 0,  marker='o', color = 'red', markerSize = 12, alpha=0.7)
    plt.plot(U_bot, 0,  marker='*', color = 'red', markerSize = 12, alpha=0.7)

    plt.axvline(strt,c='g',alpha=0.5,linewidth=4)
    plt.axvline(stop,c='r',alpha=0.5,linewidth=4)
    
    ENV = ENVELOPE.copy()
    ENV.index = ENV.index*500
    (ENV[(ENV.index > strt-2001) & (ENV.index < stop+2001)]/1000).diff().abs().plot(ax=ax,linewidth=1,color='blue',marker='x')
    (ENV[(ENV.index > strt-2001) & (ENV.index < stop+2001)]/1000).abs().plot(ax=ax,linewidth=1,color='black',marker='.')
    
#    
#    logic = 2*RISE-STOP
#    (logic).iloc[int(strt/1000):int(stop/1000)].plot(ax = ax)
#    RISE.index = RISE.index*1000
#    STOP.index = STOP.index*1000
    (RISE[(RISE.index > strt-1001) & (RISE.index < stop+1001)][RISE!=False]/100).plot(ax=ax,color='green',linewidth=0,alpha=0.6,marker='^',markerSize=25)
    (STOP[(STOP.index > strt-1001) & (STOP.index < stop+1001)][STOP!=False]/100).plot(ax=ax,color='red',linewidth=0,alpha=0.6,marker='v',markerSize=25)

    pBOT = AcousticData.iloc[strt+00:strt+2200]["/'3.051758E-5 ; 6.103516E-5 ; 6.103516E-5 ; 1.525879E-5 ; '/'S_bot'"]
    pTOP = AcousticData.iloc[strt+00:strt+2200]["/'3.051758E-5 ; 6.103516E-5 ; 6.103516E-5 ; 1.525879E-5 ; '/'S_top'"]
    
    JAM = AcousticData.iloc[strt+00:strt+2200].index
    
    pBOT = pd.DataFrame(MakePreciseEnvelope(pBOT),index=JAM) 
    pBOT = pBOT - pBOT.iloc[50]
    pTOP = pd.DataFrame(MakePreciseEnvelope(pTOP),index=JAM)
    pTOP = pTOP - pTOP.iloc[50]
    (pBOT/1000).plot(linewidth=0.5,color='red',ax=ax,alpha=0.8,secondary_y=True)
    (pTOP/1000).plot(linewidth=0.5,color='blue',ax=ax,alpha=0.8,secondary_y=True)
    
    #u = AcousticData.iloc[-500:+1000]]

    plt.xlim(strt-500,strt+4000)
    #plt.ylim(-0.05,0.05)
    plt.grid()
    plt.show()
    print(Events[i])

#%%
'''
TODO: 
    Adjust parameters to high res envelope
    Scale things down to match
    Continue the optimization
    '''
#%% Benchmarking
#
#pure = np.array([np.sin(t/10) for t in np.arange(0,12000)])
#exp = np.array([0]*3000+[-1+ np.e**(t/1000) for t in np.arange(0,1000)]+[1.7155649053185664*np.e**(-t/1000) for t in np.arange(0,6000)]+[0]*2000)
#
#event = list(pure*exp)
#event = event*4
#event = pd.DataFrame(event)
#eventS = event.shift(50).bfill()
#event = pd.concat([event,eventS],axis=1)
#event.columns = ['A','B']
##event = event.dropna()
#event.plot(alpha=0.6)
#
#def SEnvelope(AcousticData,factor = 500):  
#    OffsetFix = AcousticData - AcousticData.iloc[:80].mean()
#    Balanced = OffsetFix / OffsetFix.abs().std()
#    
#    acoustics = Balanced
#    acoustics.index = acoustics.index.values//factor
#    acoustics = acoustics.abs()
#    acoustics = acoustics.groupby(acoustics.index).mean()
#    acoustics = acoustics-acoustics.iloc[:50].mean()
#    c = acoustics.columns
#    
#    best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22
#    
#    print('envelope created')
#    return best_envelope
#
#ENVELOPE = SEnvelope(event)
#RISE = RiseMark(ENVELOPE)
#STOP = StopMark(ENVELOPE, stop_thresh = 5)
#MacroLog = EventCutter(RISE,STOP,factor=500)
#PreciseLog = SecondPass(MacroLog, event, tdms_files[0])
#print(PreciseLog)
#plt.show()
#
##
#N = pd.DataFrame(np.random.random(len(event)))/2
#N2 = pd.DataFrame(np.random.random(len(event)))/2
#
#Noise = pd.concat([N,N2],axis=1)
#Nevent = event + Noise.values
#Nevent.dropna().plot(alpha=0.6)
#
#ENVELOPE = SEnvelope(Nevent)
#RISE = RiseMark(ENVELOPE)
#STOP = StopMark(ENVELOPE, stop_thresh = 5)
#MacroLogN = EventCutter(RISE,STOP,factor=500)
#PreciseLogN = SecondPass(MacroLog, Nevent, tdms_files[0])
#print(PreciseLogN)
#plt.show()
#
##
#N = pd.DataFrame(np.random.random(len(event)))
#N2 = pd.DataFrame(np.random.random(len(event)))
#Noise = pd.concat([N,N2],axis=1)
#NNevent = event + Noise.values
#NNevent.dropna().plot(alpha=0.6)
#
#ENVELOPE = SEnvelope(NNevent)
#RISE = RiseMark(ENVELOPE)
#STOP = StopMark(ENVELOPE, stop_thresh = 5)
#MacroLogN = EventCutter(RISE,STOP,factor=500)
#PreciseLogNmax = SecondPass(MacroLog, NNevent, tdms_files[0])
#print(PreciseLogNmax)
#plt.show()
#
##
#N = pd.DataFrame(np.random.random(len(event)))*2
#N2 = pd.DataFrame(np.random.random(len(event)))*2
#Noise = pd.concat([N,N2],axis=1)
#NNNevent = event + Noise.values
#NNNevent.dropna().plot(alpha=0.6)
#
#ENVELOPE = SEnvelope(NNNevent)
#RISE = RiseMark(ENVELOPE)
#STOP = StopMark(ENVELOPE, stop_thresh = 5)
#MacroLogN = EventCutter(RISE,STOP,factor=500)
#PreciseLogNmaxsuper = SecondPass(MacroLog, NNNevent, tdms_files[0])
#print(PreciseLogNmaxsuper)
#plt.show()
#
##
#c = AcousticData.columns
#N = AcousticData.iloc[1000000:1048000]*15
#N2 = N[c[0]]
#N = N[c[1]]
#
#Noise = pd.concat([N,N2],axis=1)
#NAevent = event + Noise.values
#NAevent.dropna().plot(alpha=0.6)
#
#ENVELOPE = SEnvelope(NAevent)
#RISE = RiseMark(ENVELOPE)
#STOP = StopMark(ENVELOPE, stop_thresh = 5)
#MacroLogN = EventCutter(RISE,STOP,factor=500)
#PreciseLogNsample = SecondPass(MacroLog, NAevent, tdms_files[0])
#print(PreciseLogNsample)
#plt.show()
#
#
##
#N = pd.DataFrame(np.random.random(len(event)))/5
#N2 = pd.DataFrame(np.random.random(len(event)))/5
#Noise = pd.concat([N,N2],axis=1)
#NSevent = event + Noise.values
#NSevent.dropna().plot(alpha=0.6)
#
#ENVELOPE = SEnvelope(NSevent)
#RISE = RiseMark(ENVELOPE)
#STOP = StopMark(ENVELOPE, stop_thresh = 5)
#MacroLogN = EventCutter(RISE,STOP,factor=500)
#PreciseLogNmaxsuper = SecondPass(MacroLog, NSevent, tdms_files[0])
#print(PreciseLogNmaxsuper)
#plt.show()
#%% NOTES
'''
Immediate modification, think about using a faux start at the max of the range
^NAICE

Future modification, consider if the LF noise is really there
'''
