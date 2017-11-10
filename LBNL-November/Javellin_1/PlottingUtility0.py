'''
Plotting Utility

'''
from matplotlib import pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12,7
import numpy as np


def PlotAll(Acoustic,PreciseLog,PreciseEnvelopes,cutoff,macro_env):
    for i in np.arange(len(PreciseLog)):
        event = PreciseLog.iloc[i]
        mStart = event.Start
        mStop = event.Stop
        pStart1 = event['S_bot time']
        pStart2 = event['S_top time']
        story = event.Story
        subEventNo = event.SubEvents
        
        c = Acoustic.columns
        AcousticCut = Acoustic.iloc[mStart-1000:mStop+1000]
        MacroEnvCut = macro_env.iloc[int((mStart-1000)/1000):int((mStop+1000)/1000)]
        MacroEnvCut.index = MacroEnvCut.index*1000
        MacroEnvCut = MacroEnvCut*8
        
        ax = AcousticCut[c[0]].plot(c = 'r', alpha = 0.6)
        AcousticCut[c[1]].plot(c = 'b', alpha = 0.6, ax = ax)
        MacroEnvCut.plot(c='y',alpha=0.6,ax=ax)
        plt.plot(pStart1,0,marker='o',c='r',mec='black')
        plt.plot(pStart2,0,marker='o',c='blue',mec = 'black')
        
        pEnv = PreciseEnvelopes.iloc[i]
        A = pEnv[c[0]]
        B = pEnv[c[1]]
        
        A.plot(c = 'r')
        B.plot(c = 'blue')
        
        for i in range(len(story)):
            if i == subEventNo:
                plt.axvline(x=story[i],c='black',alpha=0.7,linewidth=4)
            else:
                plt.axvline(x=story[i],c='green',alpha=0.7,linewidth=4)
        
        plt.show()
        print('Event Start: ' + str(mStart) + ', Precise Starts: ' + str(pStart1) + ', ' + str(pStart2) + ', diff: ' + str(pStart1-pStart2))
        
        