'''
First Pass

'''
import datetime
import numpy as np
import pandas as pd

#MODIFY SO THAT EVERYONE KNOWS ALL IS COPY OKAY?

def EventCutter():
    stop.iloc[-1] = True
    Log = pd.DataFrame(columns = ['Start','Stop'])
    
    
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
    ENVELOPE= MakeEnvelope(AcousticData,factor = factor)
    RISERS = RiseMark(ENVELOPE,d1_thresh = d1_thresh, d2_thresh = d2_thresh)
    STOPS = StopMark(ENVELOPE, stop_thresh = stop_thresh)
    MACROEVENTLOG = EventCutter(RISERS,STOPS,factor = factor, pre_buffer = pre_buffer, post_buffer = post_buffer, filename=filename)
      
    print('Macro Event Log created') 
    AcousticData.index = SVDEX    
    return MACROEVENTLOG,ENVELOPE