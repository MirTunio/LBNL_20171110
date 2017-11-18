# Quench Timer 0

from nptdms import TdmsFile
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt 
from pylab import rcParams
rcParams['figure.figsize'] = 16,12

AcousticIndexes = [8,9,10,11,12,13,14,15]#[7,8,9,10,11,12,13,14,15]
VoltageIndexes = [2,3]
TriggerIndex = 0


tdms_files = glob.glob('*.tdms')
tdms_files = [tdms_files[-3]]

print(len(tdms_files))

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

#%%
  
LOG = pd.DataFrame(columns = ['File','Start','Stop'])

for file in tdms_files:

    temp = {'File':file,'Start':None,'Stop':None}
    TDDF = getDataFrame(file)
    #CARE = getTriggerCare(AcousticIndexes, TriggerIndex, TDDF)
    CARE = getVoltageCare(VoltageIndexes, TDDF)
    
    if CARE:
        temp['Start'] = CARE - 4000 #2800
        temp['Stop'] = CARE# + 10000
        
        print(file + ' LOADED')
    else:
        print('AN ERROR OCCURED LOADING: ' + file)
        continue
    
    LOG = LOG.append(temp, ignore_index=True)

def getStart(envelope):
    DFENV = pd.DataFrame(envelope)
    minima = DFENV.dropna().idxmin().values[0] #THIS IS THE PROBLEM

    return minima#np.argmin(envelope)

#%%

BESTLOG = pd.DataFrame(columns = ['Quench No.','File','Start','Stop','S1','S2','S3','S4','S5','S6','S8','S9'])


for i in np.arange(len(LOG)):
    QUENCH = LOG.iloc[i]
    FILE = QUENCH.File
    START = QUENCH.Start
    STOP = QUENCH.Stop
    
    print('Processing File: ' + FILE)
    
    temp = {'Quench No.':None,'File':FILE[:-5],'Start':START,'Stop':STOP,'S1':None,'S2':None,'S3':None,'S4':None,'S5':None,'S6':None,'S8':None,'S9':None}
    
    TDDF = getDataFrame(FILE).iloc[START:STOP]#.iloc[START-10000:STOP+10000]
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

#        plt.plot(channel_val)
#        plt.plot(trigger/10)
#        #plt.show()
#
#        plt.plot(np.array(channel_env)/420000)
#        plt.plot(trigger/10)
#        plt.show()
#        
#        print(COLS[colnum],FILE)
    BESTLOG = BESTLOG.append(temp,ignore_index=True)
BESTLOG.index = BESTLOG['Quench No.']
BESTLOG = BESTLOG.sort_values('Quench No.')

del BESTLOG['Quench No.']
BESTLOG.to_csv('BESTLOG_3.csv')

#%% DO IT AGAIN: CLOSER
UBERLOG = pd.DataFrame(columns = ['Quench No.','File','Start','Stop','S1','S2','S3','S4','S5','S6','S8','S9'])


for i in np.arange(len(BESTLOG)):
    BESTDATA = BESTLOG.iloc[i]
    FILE = BESTDATA.File+'.tdms'
    START = BESTDATA.Start
    STOP = BESTDATA.Stop
    
    print('Uber Processing File: ' + FILE)
    
    temp = {'Quench No.':None,'File':FILE[:-5],'Start':START,'Stop':STOP,'S1':None,'S2':None,'S3':None,'S4':None,'S5':None,'S6':None,'S8':None,'S9':None}
    
    
    COLS = TDDF.columns
    
    AFTBUFFER = 500
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
UBERLOG.to_csv('UBERLOG.csv')

BESTLOG = UBERLOG.copy()
#%%
def inspect(AcousticIndexes,BESTLOG):
    
    for i in np.arange(len(BESTLOG)):
        
        STARTOFFSET = 2000# 1000#60000
        STOPOFFSET = 0# 1000#4000
        
        QUENCH = BESTLOG.iloc[i]
        FILE = QUENCH.File+'.tdms'
        START = QUENCH.Start
        STOP = QUENCH.Stop
        BestStart = np.array([QUENCH.S1,QUENCH.S2,QUENCH.S3,QUENCH.S4,QUENCH.S5,QUENCH.S6,QUENCH.S8,QUENCH.S9])+START
        channels = ['S1','S2','S3','S4','S5','S6','S8','S9']
        
        TDDF = getDataFrame(FILE)
        COLS = TDDF.columns[AcousticIndexes]
        
        for i in np.arange(len(AcousticIndexes)):
            
            mid = 0
            off = 2000

    
            AUDIO = TDDF[COLS[i]].iloc[START-STARTOFFSET:STOP+STOPOFFSET]
            MARK = BestStart[i]
            
            title = FILE + ' Channel: ' +  channels[i] + ', start:' + str(MARK)
            ax = AUDIO.plot(c='blue',alpha=0.6)
            plt.plot(MARK,np.mean(AUDIO),marker='x',linewidth=0,markerSize=14,c='r')
            plt.title(title)
            
            
            
            IL = pd.rolling_mean(TDDF[TDDF.columns[2]],500)
            OL = pd.rolling_mean(TDDF[TDDF.columns[3]],500)
            VDIF = np.abs(IL-OL)
            
            VDIF.iloc[START-STARTOFFSET:STOP+STOPOFFSET].plot(ax=ax,label='VDIF')
            IL.iloc[START-STARTOFFSET:STOP+STOPOFFSET].plot(ax=ax,linewidth=2)
            OL.iloc[START-STARTOFFSET:STOP+STOPOFFSET].plot(ax=ax,linewidth=2)
            
            
            print(COLS[i])
            print(title)
            
            plt.axvline(START,c='g',alpha=0.5,linewidth=4)
            plt.axvline(STOP,c='r',alpha=0.5,linewidth=4)
            
            mid = MARK
            plt.xlim(mid-off,mid+off)
            
            plt.legend()
            plt.grid()
            #plt.savefig(FILE + '_channel_' +  channels[i] + '_start_' + str(MARK)+'.jpg')
            #plt.cla()
            #plt.clf()

            plt.show()
#                 
#            ENV = MakePreciseEnvelope(AUDIO)
#            plt.plot(ENV)
#            plt.show()
#inspect(AcousticIndexes,BESTLOG)#.iloc[:2])#.sample(2))
#inspect(AcousticIndexes,BESTLOG)      
        
#%% Inspect Voltage Triggers
#def MakePreciseEnvelope(AcousticData):
#    raw = AcousticData
#    newENV = []
#    
#    for i in np.arange(len(raw)):
#        newENV.append(AIC(raw,i,len(raw)))
#        
#    return newENV
#
#def AIC(x,k,N):
#    return k*np.log(np.var(x[1:k]))-5*(N - k - 1)*np.log(np.var(x[k+1:N]))

def inspect_voltage(AcousticIndexes,BESTLOG):
    
    for i in np.arange(len(BESTLOG)):
        
        STARTOFFSET = 2000# 1000#60000
        STOPOFFSET = 0# 1000#4000
        
        QUENCH = BESTLOG.iloc[i]
        FILE = QUENCH.File+'.tdms'
        START = QUENCH.Start
        STOP = QUENCH.Stop
        BestStart = np.array([QUENCH.S1,QUENCH.S2,QUENCH.S3,QUENCH.S4,QUENCH.S5,QUENCH.S6,QUENCH.S8,QUENCH.S9])+START
        channels = ['S1','S2','S3','S4','S5','S6','S8','S9']
        
        TDDF = getDataFrame(FILE)
        COLS = TDDF.columns[AcousticIndexes]
        
        for i in np.arange(len(AcousticIndexes)):
            
            mid = 0
            off = 2000        
            
            IL = pd.rolling_mean(TDDF[TDDF.columns[2]],500)
            OL = pd.rolling_mean(TDDF[TDDF.columns[3]],500)
            
            IL = IL-IL.dropna().iloc[:200].mean()
            OL = OL-OL.dropna().iloc[:200].mean()    
            
            VDIF = np.abs(IL-OL)
            
            ax = VDIF.iloc[START-STARTOFFSET:STOP+STOPOFFSET].plot(label='VDIF')
            IL.iloc[START-STARTOFFSET:STOP+STOPOFFSET].plot(ax=ax,linewidth=2)
            OL.iloc[START-STARTOFFSET:STOP+STOPOFFSET].plot(ax=ax,linewidth=2)
            
            MARK = BestStart[i]
            AUDIO = TDDF[COLS[i]].iloc[START-STARTOFFSET:STOP+STOPOFFSET]
            title = FILE + ' Channel: ' +  channels[i] + ', start:' + str(MARK)
            
            ENV = MakePreciseEnvelope(AUDIO)           
            ENV_DF = pd.DataFrame(ENV,index=AUDIO.index)/400000
            ENV_DF.plot(ax=ax)
             
            print(COLS[i])
            print(title)
            
            plt.axvline(START,c='g',alpha=0.5,linewidth=4)
            plt.axvline(STOP,c='r',alpha=0.5,linewidth=4)
            
            AUDIO.plot(c='blue',alpha=0.4,ax=ax,secondary_y=True)
            plt.plot(MARK,np.mean(AUDIO),marker='x',linewidth=0,markerSize=14,c='r')
            plt.axvline(MARK,c='r',linewidth=1)
            plt.title(title)            
            
            AFTBUFFER = 500
            FOREBUFFER = 500
            channel_data = TDDF[COLS[i]].iloc[MARK-FOREBUFFER:MARK+AFTBUFFER]
            channel_val = channel_data.values        
            channel_env = MakePreciseEnvelope(channel_val)  
            
            ENV_DF = pd.DataFrame(channel_env,index=channel_data.index)/200000       
            ENV_DF = ENV_DF-(pd.rolling_mean(ENV_DF,15).diff())*30
            ENV_DF.plot(ax=ax,c='black')         
#  
#            ENV_DF = pd.DataFrame(channel_env,index=channel_data.index)/400000       
#            ENV_DF = pd.rolling_mean(1000*ENV_DF.diff(),30)
#            ENV_DF.plot(ax=ax,c='purple') 
#
#            ENV_DF = pd.DataFrame(channel_env,index=channel_data.index)/400000       
#            ENV_DF = 1000*pd.rolling_mean(ENV_DF,30).diff()
#            ENV_DF.plot(ax=ax,c='orange') 
            
#            ENV_DF_2 = ((pd.DataFrame(channel_env,index=channel_data.index)/40000).diff().abs())-0.05
#            ENV_DF_2.plot(ax=ax,c='purple',alpha=0.5) 
#            
#            (ENV_DF-ENV_DF_2).plot(ax=ax,c='orange') 
            
            mid = MARK
            plt.xlim(mid-off,mid+off)
            
            plt.title(title)
            plt.legend()
            plt.grid()
            #plt.savefig('UBER' + FILE + '_channel_' +  channels[i] + '_start_' + str(MARK)+'.jpg')
            #plt.cla()
            #plt.clf()

            plt.show()
            
            #if i == 2:
            #    break
            
inspect_voltage(AcousticIndexes,BESTLOG)      

#%%
def AIC(x,k,N):
    return k*np.log(np.var(x[1:k]))+(N - k - 1)*np.log(np.var(x[k+1:N]))

def MakePreciseEnvelope(AcousticData):
    raw = AcousticData
    newENV = []
    
    for i in np.arange(len(raw)):
        newENV.append(AIC(raw,i,len(raw)))

    return newENV

def AICA(x,k,N):
    return k*np.log(np.var(x[1:k]))

def MakePreciseEnvelopeA(AcousticData):
    raw = AcousticData
    newENV = []
    
    for i in np.arange(len(raw)):
        newENV.append(AICA(raw,i,len(raw)))

    return newENV

def AICB(x,k,N):
    return (N - k - 1)*np.log(np.var(x[k+1:N]))

def MakePreciseEnvelopeB(AcousticData):
    raw = AcousticData
    newENV = []
    
    for i in np.arange(len(raw)):
        newENV.append(AICB(raw,i,len(raw)))

    return newENV

def LoadOne(filename):
     TDDF = getDataFrame(filename)
     doinconcern = TDDF.columns[10]
     print(doinconcern)
     AcousticData = TDDF[doinconcern].iloc[114825:117000]#112789  116789
     Envelope = MakePreciseEnvelope(AcousticData)
     EnvelopeA = MakePreciseEnvelopeA(AcousticData)
     EnvelopeB = MakePreciseEnvelopeB(AcousticData)
     
     Env = pd.DataFrame(Envelope,index=AcousticData.index,columns=['Envelope'])
     EnvA = pd.DataFrame(EnvelopeA,index=AcousticData.index,columns=['Part A'])
     EnvB = pd.DataFrame(EnvelopeB,index=AcousticData.index,columns=['Part B'])
     
     All = pd.concat([Env,EnvA,EnvB],axis=1)
     ax = All.plot()
 
     AcousticData = TDDF[doinconcern].iloc[112789:116789]
     Envelope = MakePreciseEnvelope(AcousticData)
     EnvelopeA = MakePreciseEnvelopeA(AcousticData)
     EnvelopeB = MakePreciseEnvelopeB(AcousticData)
     
     Env = pd.DataFrame(Envelope,index=AcousticData.index,columns=['Envelope X'])
     EnvA = pd.DataFrame(EnvelopeA,index=AcousticData.index,columns=['Part A X'])
     EnvB = pd.DataFrame(EnvelopeB,index=AcousticData.index,columns=['Part B X'])
     
     All = pd.concat([Env,EnvA,EnvB],axis=1)
     All.plot(ax = ax)
     
     
     (AcousticData*10000-10000).plot(alpha=0.5,ax=ax,color='grey')
     return None
     
LoadOne(tdms_files[0])    
     
     #400 fore aft??
     
     
     
     
     
     
