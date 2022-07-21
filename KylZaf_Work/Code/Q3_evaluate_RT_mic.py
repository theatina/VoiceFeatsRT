import pyaudio
import wave
import numpy as np
np.seterr('raise')
import matplotlib.pyplot as plt
from matplotlib import cm
from time import sleep
import pickle
import os
import pandas as pd
from threading import Thread

from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning) 

import audio_representation as au

# Stop matplotlib windows from blocking !!!! (before this line was added, frames were frozen)
plt.ion()

# algo="HistGB"
algo="LogReg"
filename = f'..{os.sep}Models{os.sep}{algo}_CalmAngry.model'
model = pickle.load(open(filename, 'rb'))

WINDOW_SIZE = 2048
CHANNELS = 1
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
user_terminated = False
#------------------------------------------------------------------------------------

# %% call back with global

def callback( in_data, frame_count, time_info, status):
    global global_block, f, fft_frame, win, spec_img, audio_blocks
    # global_block = f.readframes(WINDOW_SIZE)
    n = np.frombuffer( in_data , dtype='int16' )
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

def user_input_function():
    k = input('press "s" to terminate (then press "Enter"): ')
    if k == 's' or k == 'S':
        global user_terminated
        user_terminated = True

# %% create output stream
p = pyaudio.PyAudio()
output = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                output=False,
                input=True,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=callback)

output.start_stream()
threaded_input = Thread( target=user_input_function )
threaded_input.start()
# after starting, check when n empties (file ends) and stop'

fig = plt.figure(figsize=(12,8))
while output.is_active() and not user_terminated:
    if len( audio_blocks ) == BLOCKS2KEEP:

        # get features (mfcc profile) of audio window
        blocks_concatented = np.concatenate( audio_blocks ).astype(np.float32)/(2**15)
        au_manager.process_audio( blocks_concatented )
        mfcc_profile = au_manager.useful_mfcc_profile
        X_in = mfcc_profile.reshape(-1,20)
        
        # make predictions from training data
        preds = model.predict( X_in )
        print(repr(preds[0]))

        plt.clf()
        ax1 = fig.add_subplot(211)
        ax1.plot(mfcc_profile)

        ax2 = fig.add_subplot(212)
        # interpolation = ["hanning", "nearest"] | cmap = ["twilight", "ocean"]
        ax2.imshow(au_manager.usefull_mfcc_normalised,  interpolation='hanning', cmap=cm.twilight,  origin='lower')
        
        title_text = 'Calm'

        # store predictions for evaluation
        if (preds[0] > 0.5):
            title_text = 'Angry'

        fig.suptitle(title_text, fontsize=16)
        plt.show()
    plt.pause(0.01)


print('stopping audio')
output.stop_stream()