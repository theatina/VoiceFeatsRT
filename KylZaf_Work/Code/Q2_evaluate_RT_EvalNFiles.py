import pyaudio
import wave
import numpy as np
np.seterr('raise')
import matplotlib.pyplot as plt
from matplotlib import cm
from time import sleep
import pickle
import sys
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler

import functions as funs

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings(action='ignore', category=UserWarning) 

import audio_representation as au

# Stop matplotlib windows from blocking !!!! (before this line was added, frames were frozen)
plt.ion() 

algo="LogReg"
model_name = f"{algo}_CalmAngry"
filename = f'..{os.sep}Models{os.sep}{model_name}.model'
model = pickle.load(open(filename, 'rb'))

WINDOW_SIZE = 2048
CHANNELS = 2
RATE = 44100
FFT_FRAMES_IN_SPEC = 20

# global
# n = np.zeros(1)
global_block = np.zeros( WINDOW_SIZE*2 )
fft_frame = np.array( WINDOW_SIZE//2 )
win = np.hamming(WINDOW_SIZE)
spec_img = np.zeros( ( WINDOW_SIZE//2 , FFT_FRAMES_IN_SPEC ) )
# keep separate
#  audio blocks, ready to be concatenated
BLOCKS2KEEP = 20
audio_blocks = []
blocks_concatented = np.zeros( WINDOW_SIZE*BLOCKS2KEEP )

au_manager = au.AudioRepresentation()

#------------------------------------------------------------------------------------
if len(sys.argv)<2:
    filepath = f'..{os.sep}Data{os.sep}AudioFiles{os.sep}Angry_3rd_05{os.sep}03-01-05-02-02-01-01.wav'
else:
    filepath = sys.argv[1]

f = wave.open( filepath, 'rb' )

# %% call back with global
def callback( in_data, frame_count, time_info, status):
    global global_block, f, fft_frame, win, spec_img, audio_blocks
    global_block = f.readframes(WINDOW_SIZE)
    n = np.frombuffer( global_block , dtype='int16' )
    # begin with a zero buffer
    b = np.zeros( (n.size , CHANNELS) , dtype='int16' )
    # 0 is left, 1 is right speaker / channel
    b[:,0] = n
    # for plotting
    # audio_data = np.fromstring(in_data, dtype=np.float32)
    if len(win) == len(n):
        frame_fft = np.fft.fft( win*n )
        p = np.abs( frame_fft )*2/np.sum(win)
        fft_frame = 20*np.log10( p[ :WINDOW_SIZE//2 ] / 32678 )
        spec_img = np.roll( spec_img , -1 , axis=1 )
        spec_img[:,-1] = fft_frame[::-1]
        # keep blocks
        audio_blocks.append( n )
        while len( audio_blocks ) > BLOCKS2KEEP:
            del audio_blocks[0]
    return (b, pyaudio.paContinue)

# %% create output stream
p = pyaudio.PyAudio()
output = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                output=False,
                input=True,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=callback)

# after starting, check when n empties (file ends) and stop
predictions = []

while len(global_block) == WINDOW_SIZE*2:
    if len( audio_blocks ) == BLOCKS2KEEP:

        # get features (mfcc profile) of audio window
        blocks_concatented = np.concatenate( audio_blocks ).astype(np.float32)/(2**15)
        au_manager.process_audio( blocks_concatented )
        mfcc_profile = np.vstack( au_manager.useful_mfcc_profile )
        X_in = mfcc_profile.reshape(-1,20)

        # make predictions from training data
        preds = model.predict( X_in )
      
        title_text = 'Calm'

        # store predictions for evaluation
        if (preds[0] > 0.5):
            title_text = 'Angry'
            predictions.append(1)
        else: 
            predictions.append(0)

if "/" in filepath:
    filename = filepath.split("/")[-1]

# Class0 - calm: 02 | Class1 - angry: 05
emotion = int(filename.split("-")[2])
parts = len(predictions)
true_labels = [ 0 if emotion==2 else 1 for i in range(parts)  ]

emotion_str = "Calm" if emotion==2 else "Angry"

# perform the classifier evaluation and print/store results
funs.evaluation(parts, predictions, true_labels, emotion_str, filename)