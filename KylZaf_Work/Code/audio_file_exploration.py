import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import pyaudio
import wave
np.seterr('raise')
import matplotlib.pyplot as plt
from matplotlib import cm
from time import sleep
import pickle
import pandas as pd
import shutil


from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import audio_representation as au


def copy_audio_files(audio_file_path=r"C:\Users\Theatina\Documents\DiMasterTemp\M906\Projects\Completed\Sound\Πακέτο ασκήσεων 2 - Ήχος 1\speech_sentiment_part"):

  for cur,dir,files in os.walk(audio_file_path):
    for f in files:
      if ".wav" in f:
        emotion = str(f.split("-")[2])
        if emotion in [ "02", "05" ]:
          src = os.path.join(cur,f)
          dst = f"..{os.sep}Data{os.sep}AudioFiles"
          shutil.copy(src, dst)


def explore_audio():
  
  WINDOW_SIZE=2048
  CHANNELS=2

  f = wave.open( f'..{os.sep}Data{os.sep}AudioFiles{os.sep}Angry_3rd_05{os.sep}03-01-05-02-02-01-01.wav', 'rb' )
  global_block = f.readframes(WINDOW_SIZE)
  n = np.frombuffer( global_block , dtype='int16' )
  b = np.zeros( (n.size , CHANNELS) , dtype='int16' )


def trim_pauses():
  pass