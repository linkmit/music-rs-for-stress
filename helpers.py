import librosa
import numpy as np
import scipy.linalg
import scipy.stats

# from https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd
def ks_key(X):
    '''Estimate the key from a pitch class distribution
    
    Parameters
    ----------
    X : np.ndarray, shape=(12,)
        Pitch-class energy distribution.  Need not be normalized
        
    Returns
    -------
    major : np.ndarray, shape=(12,)
    minor : np.ndarray, shape=(12,)
    
        For each key (C:maj, ..., B:maj) and (C:min, ..., B:min),
        the correlation score for `X` against that key.
    '''
    X = scipy.stats.zscore(X)
    
    # Coefficients from Kumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    major = scipy.stats.zscore(major)
    
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    minor = scipy.stats.zscore(minor)
    
    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)
    
    return major.T.dot(X), minor.T.dot(X)

# adapted from: 
# https://medium.com/@oluyaled/detecting-musical-key-from-audio-using-chroma-feature-in-python-72850c0ae4b1
def estimate_tonic_mode(y_har,sr):
   chromagram = librosa.feature.chroma_stft(y=y_har, sr=sr)
   mean_chromagram = chromagram.mean(axis=1)
   major_scores, minor_scores = ks_key(mean_chromagram)
   all_scores = np.concatenate((major_scores, minor_scores))
   best_key_index = np.argmax(all_scores)
   tonics = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
   modes = ['Major', 'Minor']
   key = tonics[best_key_index % 12]
   mode = modes[best_key_index // 12]
   return key, mode

