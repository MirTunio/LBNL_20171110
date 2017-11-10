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
    #IL = pd.rolling_mean(tddf[columns[VoltageIndex]],5000)
    IL = tddf[columns[VoltageIndex]]
    care = np.nonzero(IL < voltage_threshold)[0][0]
    return care

#%% FirstPass Wolf Edition
def MacroEnvelope(AcousticData,factor = 1000):  
    OffsetFix = AcousticData - AcousticData.iloc[:80000].mean()
    Balanced = OffsetFix / OffsetFix.abs().std()
    
    acoustics = Balanced
    acoustics.index = acoustics.index.values//factor
    acoustics = acoustics.abs()
    acoustics = acoustics.groupby(acoustics.index).mean()
    acoustics = acoustics-acoustics.iloc[:50].mean()
    c = acoustics.columns
    
    #best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22
    best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22
    
    print('envelope created')
    #best_envelope.index = best_envelope.index*factor
    return best_envelope #[acoustics,best_envelope,Balanced]


#AcousticData.iloc[1500000:2000000].plot(alpha=0.6)
#MakeEnvelope(AcousticData.iloc[:500000])[1].plot(alpha=0.6)
#x = MacroEnvelope(AcousticData,factor=1000000)[2].plot(alpha=0.6)


#%%MODULE MODULE Load Acoustics Wolf Edition
def RiseMark(env,d1_thresh = 7, d2_thresh = 10): #12
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
    
    T.index = T.index + 1
    return T
    
def StopMark(env, stop_thresh = 15):
    diff = np.abs(env)#.diff())
    d1 = diff < stop_thresh
    d2 = diff < stop_thresh
    d3 = diff < stop_thresh
    
    d2.index = d2.index+1
    d3.index = d2.index+2
    
    T = (d1 & d2) & d3
    
    T2 = T.tail(len(T)-1)
    T2.index = T2.index-1
    T = T2&(T^T2)
    
    c = T.columns
    T = T[c[0]]|T[c[0]]
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
    #gentime = str(datetime.datetime.now())[:-7].replace(':','_').replace(' ','  ').replace('-','_')
    #Log.to_csv(filename + ' ' + gentime + '.csv') 
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

def SecondPass(MacroEventLog,AcousticData,filename):
    PreciseLog = pd.DataFrame(columns = ['S_bot', 'S_top', 'S_top - S_bot'])
    dex = MacroEventLog.index
    C = AcousticData.columns
    
    for i in np.arange(len(MacroEventLog)):
        vent = MacroEventLog.iloc[i]
        strt = vent.Start
        stop = vent.Stop
        
        EntropyEnvelopeA = MakePreciseEnvelope(AcousticData[C[0]][strt-1000:strt+2000]) 
        EntropyEnvelopeB = MakePreciseEnvelope(AcousticData[C[1]][strt-1000:strt+2000])
        
        precise_A = getStart(EntropyEnvelopeA)
        precise_B = getStart(EntropyEnvelopeB)
        
        temp = {'S_bot':precise_A, 'S_top':precise_B, 'S_top - S_bot':precise_B-precise_A}
        
        PreciseLog = PreciseLog.append(temp, ignore_index=True)
        
    PreciseLog.index = dex
    PreciseEventLog = pd.concat([MacroEventLog,PreciseLog],axis=1)

    print('Precise log generated')
    return PreciseEventLog

def ThirdPass(PreciseLog, AcousticData, filename):
    UBERLOG = pd.DataFrame(columns = ['Quench No.','File','Start','Stop','S_top','S_bot'])

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
        UBERLOG.index.name = 'Quench No.'
        
    return UBERLOG

#%% AND NOW WE COMPARE...






#%%
#timebins = np.append(np.arange(0,len(Deltas),235),len(Deltas)-1)
#
#ax = plt.axes()#Deltas.hist(alpha=1)
#for i in np.arange(len(timebins)-1):
#    timewise = Deltas.iloc[timebins[i]:timebins[i+1]]
#    timewise.hist( color= (i/(len(timebins)),0,0), alpha=0.3,bins=200)
#    plt.xlim(-2000,2000)
#    plt.show()
    





