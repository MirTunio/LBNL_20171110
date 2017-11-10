'''
First Pass

'''
import datetime
import numpy as np
import pandas as pd


def MakeEnvelope(acoustics,factor = 1000):
    acoustics.index = acoustics.index.values//factor
    acoustics = acoustics.abs()
    acoustics = acoustics.groupby(acoustics.index).mean()
    acoustics = acoustics - np.mean(acoustics)
    
    c = acoustics.columns
    best_envelope = pd.DataFrame(acoustics[c[0]].values*acoustics[c[1]].values)*22

    print('envelope created')
    return best_envelope


def RiseMark(env,d1_thresh = 0.00020, d2_thresh = 0.0015):
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
    
    print('risers found')
    return T

def StopMark(env, stop_thresh = 0.00020):
    diff = np.abs(env.diff())
    d1 = diff < stop_thresh
    d2 = diff < stop_thresh
    d3 = diff < 5000 #stop_thresh
    
    d2.index = d2.index+1
    d3.index = d2.index+2
    
    T = d1 & d2
    
    T2 = T.tail(len(T)-1)
    T2.index = T2.index-1
    T = T2&(T^T2)
    
    c = T.columns
    T = T[c[0]]|T[c[0]]
    
    print('stops found')
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
    gentime = str(datetime.datetime.now())[:-7].replace(':','_').replace(' ','  ').replace('-','_')
    Log.to_csv(filename + ' ' + gentime + '.csv') 
    print('Events cut')
    return Log     
            
    
def FirstPass(AcousticData, factor = 1000, d1_thresh = 0.0020, d2_thresh = 0.0015, stop_thresh = 0.00020, pre_buffer = 1000, post_buffer = 1000, filename='TEST'):
    SVDEX = AcousticData.index
    ENVELOPE = MakeEnvelope(AcousticData,factor = factor)
    RISERS = RiseMark(ENVELOPE,d1_thresh = d1_thresh, d2_thresh = d2_thresh)
    STOPS = StopMark(ENVELOPE, stop_thresh = stop_thresh)
    MACROEVENTLOG = EventCutter(RISERS,STOPS,factor = factor, pre_buffer = pre_buffer, post_buffer = post_buffer, filename=filename)
      
    print('Macro Event Log created') 
    AcousticData.index = SVDEX    
    return MACROEVENTLOG,ENVELOPE