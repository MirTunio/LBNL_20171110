import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7,7
from nptdms import TdmsFile

'''
> Load file
> Generate envelope
> Generate novel rise and stop marks (EXPLICIT PARAMS)
> Generate neat output of times (index, power etc.)
'''

files = glob.glob('*.tdms')
filE = files[0]
tdms_file = TdmsFile(filE)
tddf = tdms_file.as_dataframe()

#def GetAcoustics(tddf):
#	trigger_threshold = 0.3
#	cols = tddf.columns
#	
#	#Accessing dataset	
#	trigger_channel = np.abs(tddf[cols[0]]) # set index to channel with 
#	care = np.nonzero(trigger_channel > trigger_threshold)[0][0]
#	acoustics = tddf[[cols[2],cols[3]]].head(care)  ##,cols[3]
#	print('file loaded successfully')
#	return acoustics

def GetAcoustics(tddf):
	trigger_threshold = 0.3
	cols = tddf.columns
	
	#Accessing dataset	
	trigger_channel = np.abs(tddf[cols[0]]) # set index to channel with 
	care = 110000000#np.nonzero(trigger_channel > trigger_threshold)[0][0]
	acoustics = tddf[[cols[1],cols[2]]].head(care)  ##,cols[3]
	print('file loaded successfully')
	return acoustics

def MakeEnvelope(acoustin,factor = 1000):
    envelope_factor = factor
    acoustin.index = acoustin.index.values//envelope_factor
    acoustin = acoustin.abs()
    acoustin = acoustin.groupby(acoustin.index).mean()
    acoustin = acoustin - np.mean(acoustin) #removing offset - will need modification ...
    
    c = acoustin.columns
    
    sums = (acoustin[c[0]].values*acoustin[c[1]].values)*22
    acoustin[c[0]] = sums
    acoustin[c[1]] = sums
    print('envelope created')
    return acoustin

def MakeEnvelopeOLD(acoustin,factor = 1000):    
	 envelope_factor = factor
	 acoustin.index = acoustin.index.values//envelope_factor
	 acoustin = acoustin.abs()
	 acoustin = acoustin.groupby(acoustin.index).mean()
	 acoustin = acoustin - np.mean(acoustin) #removing offset - will need modification ...
	 print('envelope created')
	 return acoustin
	
def RiseMark_broadcast_df_new(env):
    dif = env.diff()
    d1 = (dif > 0.00020)
    d1.index = d1.index + 1
    
    dif = env.diff()
    d2 = (dif < 0.0015)
    d2.index = d1.index + 1	
    	
    T = d1 & d2	
    T.index = T.index - 2
    	
    c = T.columns
    T = T[c[0]]&T[c[1]]
    print('risers found')
    
    T2 = T.tail(len(T)-1)
    T2.index = T2.index-1
    T = T2&(T^T2)
    
    return T

def StopMark_broadcast_df_new(envelope):
    diff = np.abs(env.diff())
    d1 = diff < 0.00020
    d2 = diff < 0.00020
    d3 = diff < 0.00020
    
    d2.index = d2.index+1
    d3.index = d2.index+2
    
    T = d1 & d2# & d3
    
    c = T.columns
    
    T2 = T.tail(len(T)-1)
    T2.index = T2.index-1
    T = T2&(T^T2)
    
    T = T[c[0]]|T[c[1]]
    print('stops found')
    
    return T
       
	
def event_cutter(rise,stop,factor = 1000):
    stop.iloc[-1] = True
    Log = pd.DataFrame(columns = ['Start','Stop','SubEvent No.'])
    State = False
    logic = 2*rise - stop
    
    temp = {'Start':None,'Stop':None,'SubEvent No.':None}
    subno = 1
    for i in np.arange(len(logic)):
        here = logic.iloc[i]
            
        if State == True and here > 0:
            subno +=1
            
        elif State == False and here > 0:
            State = True
            temp['Start'] = i*factor - 1000
        
        elif State == True and here < 0:
            temp['Stop'] = i*factor + 1000
            temp['SubEvent No.'] = subno
            Log = Log.append(temp, ignore_index=True)
            temp = {'Start':None,'Stop':None,'SubEvent No.':None}
            subno = 1
            State = False
        
        Log.index.name = 'Event No.'
    Log.to_csv('Log_newera_1.csv')
    return Log	


def new_cutter(acoustics,risers,factor = 1000):
	starts = []
	stops = []
	risers = np.nonzero(risers)[0]
	
	for rise in risers:
		time = rise*factor
		start = time-1000
		end = time+8000
		
		starts.append(start)
		stops.append(end)
		
	print('events cut out')
	return starts,stops

def event_plot_purist(starts,stops,acoust):
    sample_frequency = 1000 #1000 Smaples = 1ms
    starts = np.array(starts)/sample_frequency
    stops =  np.array(stops)/sample_frequency
    acoust.index = acoust.index/sample_frequency
    #acoust.plot(c='grey',alpha=0.3)
    for i in np.arange(len(starts)):
        print(str(i),starts[i],stops[i])
        acoust[starts[i]:stops[i]].plot(legend=None)
        plt.xlabel('time /ms')
        plt.ylabel('Transducer Voltage / V')
        #plt.savefig(str(i))
        #plt.clf()
        plt.show()

def event_plot_purist(frame_sample,acoust):
    starts = frame_sample.Start
    stops = frame_sample.Stop
    vents = frame_sample[frame_sample.columns[2]]
    
    for i in frame_sample.index:
        print(str(i),starts[i],stops[i])
        acoust[starts[i]:stops[i]].plot(legend=None)
        plt.xlabel('time /ms')
        plt.ylabel('Transducer Voltage / V')
        plt.title('Event No.: '+ str(i) + ', '+ 'Start: ' + str(starts[i]) + ', '+'Stop: ' + str(stops[i]) + ', ' + 'No. Sub-Events: '+ str(vents[i]))
        #plt.savefig(str(i))
        #plt.clf()
        plt.show()
        

def formax(risers,factor = 1000):
    risers = np.nonzero(risers)[0]
    Log = pd.DataFrame(columns = ['Start','Stop','SubEvents'])
    i = 0
    for rise in risers:
        temp = {'Start':rise*factor-1000,'Stop':rise*factor+8000, 'SubEvents':0}        
        Log = Log.append(temp, ignore_index=True)	
        i += 1
	 
    Log.index.name = 'Event No.'
    Log.to_csv('Datalog_SAMPLE_3.csv')
    print('log file created')
    return(Log)

#------------LOG TESTS START-------------
acoust = GetAcoustics(tddf)
env = MakeEnvelope(acoust)
MarkRise = RiseMark_broadcast_df_new(env)
MarkStop = StopMark_broadcast_df_new(env)
log = event_cutter(MarkRise,MarkStop)
acoust = GetAcoustics(tddf)
oldenv = MakeEnvelopeOLD(acoust)
# 
#acoust = GetAcoustics(tddf) 
#env.index = env.index
#ax = acoust.plot(c='grey',legend=False) 
#env#.index = env.index*1000
#select = env*100
#select.index = select.index*1000
#select.plot(ax=ax,marker='x')
##plt.show()
#for i in np.arange(len(log)):
#    start = log.iloc[i].Start
#    stop = log.iloc[i].Stop
#    #oldenv[int(start/1000):int(stop/1000)].plot(c='red',ax = ax, linewidth = 4, alpha = 0.5,legend=False)
#    acoust[int(start):int(stop)].plot(c='red',ax = ax, linewidth = 4, alpha = 0.5,legend=False)
#plt.show() 
     
#------LOG TEST END---------------------

#--------OUTPUT FINALE----------------------
#acoust = GetAcoustics(tddf)
#env = MakeEnvelope(acoust)
#MarkRise = RiseMark_broadcast_df_new(env)
#MarkStop = StopMark_broadcast_df_new(env)
#event_cutter(MarkRise,MarkStop)
#Log = pd.read_csv('Log_SAMPLE_DATA.csv')
#event_plot_purist(Log,GetAcoustics(tddf))
#============================================
#----------------COMMENT-----------------
#####-----------TESTING---------
#acoust = GetAcoustics(tddf)	
#env = MakeEnvelope(acoust)
##risers = RiseMark_broadcast_df_new(env)
##acoust = GetAcoustics(tddf)	
##starts,stops = new_cutter(acoust,risers)
##print(len(starts))
###event_plot(starts,stops,acoust)		
#c = env.columns
###formax(risers)
##		
###------------Mark Diagnostics------RISE-----------
#MarkRise = RiseMark_broadcast_df_new(env)
#MarkStop = StopMark_broadcast_df_new(env)
#
##Mark = 2*MarkRise-MarkStop/100
#Mark = RiseMark_broadcast_df_new(env)/300
##env = MakeEnvelopeOLD(acoust)
#
#env[c[1]].plot(c='red',alpha=0.3)
##env.diff()[c[1]][:].plot(c='blue',marker='x')
#plt.ylim(-0.005,0.020)
#plt.plot(Mark[:],c='g',linewidth=7,alpha=0.3,marker='o')
##plt.show()
#
#
#env[c[0]].plot(c='black',alpha=0.4)
##env.diff()[c[0]][:].plot(c='blue',marker='x')
#plt.ylim(-0.005,0.020)
#plt.plot(Mark[:],c='g',linewidth=7,alpha=0.3,marker='o')
#
##plt.show()
##
###------------Mark Diagnostics-----STOP-----------
##
#Mark = StopMark_broadcast_df_new(env)/300
#
#env[c[1]].plot(c='black',alpha=0.4)
##env.diff()[c[1]][:].plot(c='blue',marker='x')
#plt.ylim(-0.005,0.020)
#plt.plot(Mark[:],c='r',linewidth=7,alpha=0.3,marker='o')
##plt.show()
#
#env[c[0]].plot(c='black',alpha=0.4)
#env.diff()[c[0]][:].plot(c='blue',marker='x')
#plt.ylim(-0.005,0.020)
#plt.plot(Mark[:],c='r',linewidth=7,alpha=0.3,marker='o')
#plt.show()
#------UNCOMMENT-------------
###--------------OG2-------RISE-----------
##env[c[1]][int(2180/2):int(2220/2)].plot(c='grey',marker='x',grid=True)
##env.diff()[c[1]][int(2180/2):int(2220/2)].plot(c='blue',marker='x')
##plt.ylim(-0.010,0.020)
##plt.plot((RiseMark_broadcast_df_new(env)[c[1]]/100)[int(2180/2):int(2220/2)],c='g',linewidth=7,alpha=0.3,marker='o')
##plt.grid(which='both')
##plt.show()
##
##env[c[0]][int(2180/2):int(2220/2)].plot(c='grey',marker='x',grid=True)
##env.diff()[c[0]][int(2180/2):int(2220/2)].plot(c='blue',marker='x')
##plt.ylim(-0.010,0.020)
##plt.plot((RiseMark_broadcast_df_new(env)[c[0]]/100)[int(2180/2):int(2220/2)],c='g',linewidth=7,alpha=0.3,marker='o')
##plt.grid(which='both')
##plt.show()
#
#
#
#
##---------NOTES---------------
#'''
#Fix Rise to find in betweeners
#Fix Stops
#Implement OG event cutter
#flesh out output stub
#win.
#'''
