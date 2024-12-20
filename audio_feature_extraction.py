import librosa
import pandas as pd
import numpy as np
import helpers
from glob import glob
from helpers import *
import os

DIRECTORY = "MySongsForStressRegulation/"
files = glob(DIRECTORY + "*.mp3")

df_list = []

for file in files:
    y, sr = librosa.load(file, sr=None)
    
    # Note: reduce features
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    key, mode = helpers.estimate_tonic_mode(y_harmonic, sr)
    rms = librosa.feature.rms(y=y).mean(axis=1)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean(axis=1)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(axis=1)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)[0]
    flatness = librosa.feature.spectral_flatness(y=y).mean(axis=1)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean(axis=1)[0]
    
    file_features = {
        'file': os.path.basename(file),
        'tempo': tempo,
        'key': key,
        'mode': mode,
        'rms': rms,
        'centroid': centroid,
        'bandwidth': bandwidth,
        'contrast': contrast,
        'flatness': flatness,
        'rolloff': rolloff
    }
    
    df_list.append(file_features)
    print(df_list)

df = pd.DataFrame(df_list)

print(df)

df.to_csv('audio_features.csv', index=False)  