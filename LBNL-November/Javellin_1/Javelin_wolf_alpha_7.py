#%% IMPORT required libraries
from nptdms import TdmsFile # Opens TDMS files 
import numpy as np # Mathematics and efficient array operations library
import pandas as pd # Data table algebra library
import glob # Utility to read filenames of files in subfolder
from matplotlib import pyplot as plt # Plotting utility
from pylab import rcParams # Plotting utility additional function to change plot size
rcParams['figure.figsize'] = 16,12 # Defining plot size in inches
import os # Interface with operating system
import datetime # For accessing current date from system

#%% Load data into pandas dataframe using nptdms
# input: single filename
# output: dataframe

def getDataFrame(filename):
    if type(filename) == str:
        tdms_file = TdmsFile(filename)
        tddf = tdms_file.as_dataframe()
    else:
        raise TypeError('I need a single filename')
    return tddf

#%% Determine index at which quench starts using threshold on the voltage measurement 
# input:
#    VoltageIndex = python list of indexes where the voltage measurements are in dataframe
#    tddf = dataframe from 'getDataFrame'
# output: 
#    index of the start of quench

def getVoltageCare(VoltageIndex,tddf):
    columns = tddf.columns
    voltage_threshold = 0.400
    
    IL = tddf[columns[VoltageIndex]].abs() #Absolute value of voltage signal
    care = np.nonzero(IL > voltage_threshold)[0][0] #locates index where voltage crosses threshold
    return care

#%% Generate macro envelope, downsamples by factor value using mean. Also removes offsets and matches amplitudes
# input:
#    AcousticData: dataframe with only columns of acoustic data cut before the quench
#    factor: The factor by which to downsample, default is 500:1
# output: 
#    Envelope of acoustic data

def MacroEnvelope(AcousticData,factor = 500):  
    #removing offset and balancing amplitudes
    OffsetFix = AcousticData - AcousticData.iloc[:80000].mean()
    Balanced = OffsetFix / OffsetFix.abs().std()
    
    #downsampling
    acoustics = Balanced
    acoustics.index = acoustics.index.values//factor
    acoustics = acoustics.abs()
    acoustics = acoustics.groupby(acoustics.index).mean()
    acoustics = acoustics-acoustics.iloc[:50].mean()
    
    #multiplying envelopes of both channels, result is sensitive to changes which occur only on both channels
    c = acoustics.columns
    best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22 #factor of 22 is chosen arbitrarily to make numbers easier to deal with
    
    print('envelope created')
    return best_envelope

#%% Find indexes of at which the signal is 'rising'. Passes a 3 index kernel window over envelope
#   which is sensitive to start of acoustic events 
# input:
#    env: macro envelope
#    d1_thresh: The amount it rises after potential riser "  ./  "
#    d2_thresh: TThe quiet it must see before the potential riser "  _. "
# output: 
#    dataframe with indexes of all the risers

def RiseMark(env,d1_thresh = 12, d2_thresh = 5): #12
    '''
    riser anatomy: _./
    d1 thresh: The amount it rises after potential riser "  ./  "
    d2 thresh: The quiet it must see before the potential riser " _. "
    
    KERNEL: [d2][.][d1]
    '''

    dif = env.diff() #computes differences between envelope points
    d1 = (dif > d1_thresh) #Identifies which points exceed d1 threshold in difference
    d1.index = d1.index + 1  #Adds 1 to index value
    
    dif = env.diff()
    d2 = (dif < d2_thresh) #Identifies points which are below d2 threshold in difference
    d2.index = d1.index + 1	#Adds 1 to index of d1, effectively adds 2 to index
    	
    T = d1 & d2	#performs 'AND' operation between d1 and d2 along matching indicies
    T.index = T.index - 2 #resets index

    c = T.columns
    T = T[c[0]]&T[c[0]] #perfroms'AND' operation between both channels, returns only where both channels rise
    
    #Removes consective 'rise' marks, keeping only earliest
    T2 = T.tail(len(T)-1)
    T2.index = T2.index-1
    T = T2&(T^T2)
    
    T.index = T.index + 2 # Reset index to match signal
    return T

#%% Find indexes of at which the signal is below noise threshold.Passes a 3 index kernel window over envelope
#   which is sensitive to 'quiet' portions of signal
# input:
#    env: macro envelope
#    stop_thresh: The noise level, as measured by macro envelope
# output: 
#    dataframe with indexes of all the stop locations

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

#%% Cuts out events using start and stop dataframes. Function loops through RiseMark and StopMark 
#   values and tracks a state variable which indicates whether or not it is looping over a region
#   where an event has started. If state variable is FALSE and RISE is encountered, a 'start' value
#   is added to the log and state is made TRUE. If while state variable is TRUE another RISE is enc-
#   ountered, it marks a subevent. If state is TRUE and encounters a STOP, the function logs the stop
# input:
#    rise: dataframe from RiseMark
#    stop: dataframe from StopMark
#    factor: the downsampling factor
#    pre_buffer: padding before event to add into log
#    post_buffer: padding after event to add into log
#    filename: name of file, added into log file
# output: 
#    dataframe with rough start and stop indexes of each event

def EventCutter(rise,stop,factor = 500, pre_buffer = 1000, post_buffer = 1000, filename='TEST'):
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
            #dfseries = AcousticData.iloc[temp['Start']: temp['Stop']]
            #test = AHenv(dfseries[dfseries.columns[1]])
            
            if True:#test > 17:
                story.append(i*factor + post_buffer)
                temp['Stop'] = i*factor + post_buffer
                temp['SubEvents'] = subno
                temp['Story'] = np.array(story)        
                Log = Log.append(temp, ignore_index=True)
            
            
                temp = {'Start':None,'Stop':None,'SubEvents':None, 'Story':None}
                subno = 1
                story = []
                State = False
            else:
                temp = {'Start':None,'Stop':None,'SubEvents':None, 'Story':None}
                subno = 1
                story = []
                State = False
            
    Log.index.name = 'Event No.'
    return Log  

#%% Generates AIC mapping
# input:
#    AcousticDataIN: Acoustic data to map
# output: 
#    python list, mapping of acoustic data

def MakePreciseEnvelope(AcousticDataIN):
    newENV = []
    L = len(AcousticDataIN)
    for i in np.arange(L):
        newENV.append(AIC(AcousticDataIN,i,L))
    return newENV

#%% AIC function computed on each data point of acoustic window passed in
# input:
#    x: Window of acoustic data
#    k: Index of point in the window
#    N: Length of window
# output: 
#    AIC mapped value of point

def AIC(x,k,N):
    return k*np.log(np.var(x[1:k]))+(N - k - 1)*np.log(np.var(x[k+1:N]))

#%% Locates minima in AIC mapping
# input:
#    envelope: AIC mappend acoustic window
# output: 
#    index of the accurate start time
def getStart(envelope):
    DFENV = pd.DataFrame(envelope)
    minima = DFENV.dropna().idxmin().values[0]
    return minima

#%% Loops over start and stop values from MacroEventLog (above) to obtain precise start times for each channel
#   Uses an 'Akaike Information Criteria' (AIC) mapping, a measure of entropy at each point in the window relative 
#   to all other points in the window
# input:
#    MacroEventLog: MacroLog from above
#    AcousticData: Full length of acoustic data
#    filename: name of file to add to the log
# output: 
#    dataframe with indexes of all preicise start times + macrolog dataframe
#    NOTE: the precise start times are relative to the start time from macro log

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

#%% Essentially repeats what SecondPass does, except uses accurate start times instead of start times from
#   the macrolog
# input:
#    PreciseLog: PreciseLog from above
#    AcousticData: Full length of acoustic data
#    filename: name of file to add to the log
# output: 
#    dataframe with indexes of all very preicise start times, along with macro start and stop times
#    NOTE: the very precise start times are relative to the start time from macro log

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

#%% Additional function added to reduce logging of low frequency noise as events
def AHenv(dfseries):
    factor=50
    dex = dfseries.index
    dfseries.index = dfseries.index.values//factor
    dfseries = dfseries.groupby(dfseries.index).std()
    dfseries = dfseries-dfseries.mean()
    filt = dfseries
    if ((sum(filt.abs())/len(filt))*10000) < 20:
        return False
    else:
        return True

###################################################################################
#%% USAGE
AcousticIndexes = [1,2] #S_top and S_bot
VoltageIndex = 3 #IL
tdms_files = glob.glob('*.tdms') #finding tdms files in folder

TDDF = getDataFrame(tdms_files[0]) #loading data to memory
COLS = TDDF.columns #obtaining column names
CARE = getVoltageCare(VoltageIndex, TDDF) #obtaining index of quench start
AcousticData = TDDF[COLS[[AcousticIndexes]]].iloc[:CARE] #obtaining acoustic data from before the quench
ENVELOPE = MacroEnvelope(AcousticData) #creating envelope
RISE = RiseMark(ENVELOPE) #getting risers
STOP = StopMark(ENVELOPE, stop_thresh = 5) #getting stops

MacroLog = EventCutter(RISE,STOP,factor=500) #creating macrolog

#%%IGNORE, for plotting later...
RISE.index = RISE.index*500
STOP.index = STOP.index*500
#%% Adding column to macrolog to filter out events whichare actually LF noise
gem = []
for i in np.arange(len(MacroLog)):
    event = MacroLog.iloc[i]
    strt = event.Start
    stop = event.Stop
    acoust = AcousticData.iloc[strt:stop]
    acoust = acoust[acoust.columns[0]]
    gem += [AHenv(acoust)]
    #print(gem)

MacroLog['Validate'] = gem #adds Validation column to log

#%%
LogNew = MacroLog[MacroLog.Validate].copy() #Selects events with intensity in HF part of spectrum
PreciseLog = SecondPass(LogNew.sample(5), AcousticData, tdms_files[0]) #creating precise log, sampling 5 points here for testing purposes
UberLog = ThirdPass(PreciseLog, AcousticData, tdms_files[0]) #creating very precise log

UberLog.to_csv('EVENTLOG_20171106_LASTWEEK.csv')


#######################################################################
#%% PLOTTING SAMPLE OF RESULTS

SVLOG = UberLog.copy()
UberLog = SVLOG.sample(3)

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
    
    cutten = AcousticData.iloc[strt-1000:stop+3000]
    ax = cutten.plot(alpha=0.6)
    
    #S_top is BLUE, S_bot is ORANGE
    plt.plot(P_top, 0,  marker='o', color = 'blue', markerSize = 12, alpha=0.7)
    plt.plot(U_top, 0,  marker='*', color = 'blue', markerSize = 12, alpha=0.7)
    
    plt.plot(P_bot, 0,  marker='o', color = 'red', markerSize = 12, alpha=0.7)
    plt.plot(U_bot, 0,  marker='*', color = 'red', markerSize = 12, alpha=0.7)

    plt.axvline(strt,c='g',alpha=0.5,linewidth=4)
    plt.axvline(stop,c='r',alpha=0.5,linewidth=4)
    
    ENV = ENVELOPE.copy()
    ENV.index = ENV.index*500
    (ENV[(ENV.index > strt-2001) & (ENV.index < stop+2001)]/1000).diff().abs().plot(ax=ax,linewidth=1,color='blue',marker='x', label = 'MACRO ENVELOPE DIFFERENCES')
    (ENV[(ENV.index > strt-2001) & (ENV.index < stop+2001)]/1000).abs().plot(ax=ax,linewidth=1,color='black',marker='.',label='MACRO ENVELOPE')
    
#    
#    logic = 2*RISE-STOP
#    (logic).iloc[int(strt/1000):int(stop/1000)].plot(ax = ax)
#    RISE.index = RISE.index*1000
#    STOP.index = STOP.index*1000
    (RISE[(RISE.index > strt-1001) & (RISE.index < stop+1001)][RISE!=False]/100).plot(ax=ax,color='green',linewidth=0,alpha=0.6,marker='^',markerSize=25, label = 'RISERS')
    (STOP[(STOP.index > strt-1001) & (STOP.index < stop+1001)][STOP!=False]/100).plot(ax=ax,color='red',linewidth=0,alpha=0.6,marker='v',markerSize=25, label = 'STOPS')

    pBOT = AcousticData.iloc[strt+00:strt+2200]["/'3.051758E-5 ; 6.103516E-5 ; 6.103516E-5 ; 1.525879E-5 ; '/'S_bot'"]
    pTOP = AcousticData.iloc[strt+00:strt+2200]["/'3.051758E-5 ; 6.103516E-5 ; 6.103516E-5 ; 1.525879E-5 ; '/'S_top'"]
    
    JAM = AcousticData.iloc[strt+00:strt+2200].index
    
    pBOT = pd.DataFrame(MakePreciseEnvelope(pBOT),index=JAM) 
    pBOT = pBOT - pBOT.iloc[50]
    pTOP = pd.DataFrame(MakePreciseEnvelope(pTOP),index=JAM)
    pTOP = pTOP - pTOP.iloc[50]
    (pBOT/1000).plot(linewidth=0.5,color='red',ax=ax,alpha=0.8,secondary_y=True, label='S_bot preceise envelope')
    (pTOP/1000).plot(linewidth=0.5,color='blue',ax=ax,alpha=0.8,secondary_y=True, label = 'S_top precise envelope')
    
    #u = AcousticData.iloc[-500:+1000]]

    #plt.xlim(strt-500,strt+4000)
    #plt.ylim(-0.05,0.05)
    plt.grid()
    #plt.show()

    
    cutten = AcousticData.iloc[strt:stop]#cutten[cutten.columns[1]]
    
    def AHenv(dfseries,factor=50):
        dex = dfseries.index
        dfseries.index = dfseries.index.values//factor
        dfseries = dfseries.groupby(dfseries.index).std()
        dfseries = dfseries-dfseries.mean()
        filt = dfseries
        return (sum(filt.abs())/len(filt))*10000
    
    cutten = cutten[cutten.columns[1]]
    filt = AHenv(cutten)
    print(filt)##filt.plot()
    
    plt.ylim()#-0.005,0.020)
    plt.legend()
    plt.show()
    
#    cutten.diff().hist()
    print(Events[i])
    print()





















