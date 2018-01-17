""" The entry points of the identification for telecom mobile voice.

    Author: Sean Wu, Bill Haung
    NCU CSIE 3B, Taiwan
"""

import os
from os.path import join, isfile
import pickle
import sys
import subprocess
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from patterns_cmp import ptns_cmp

def televoice_identify(filepath, threshold=None, multiproc=True):
    """ Calculate the difference indices between target audio and each golden audio wavfiles.

    Parameters
    ----------

    Return
    ------
    The dictionary containing the difference indices between target audio and each golden audio
      wavfiles.
    """

    golden_ptns = read_golden_ptns(join("golden_wav"))     # load golden wavfiles

    # Call the ffmpeg to convert (normalize) the input audio into:
    #    sample rate    8000 Hz
    #    bit depth      16
    #    channels       mono (merged)
    try:
        subprocess.call(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'panic', '-i', filepath,
                         '-ac', '1', '-ar', '8000', '-sample_fmt', 's16', '-f', 'wav',
                         'converted.tmp'])
    except FileNotFoundError:
        print("[Error] Require ffmpeg to convert the audio in sepcific format.")
        sys.exit(2)    # ffmpeg require
    (rate, sig) = wav.read("converted.tmp")    # read the target wavfile
    target_mfcc = mfcc(sig, rate, appendEnergy=False)
    result = ptns_cmp(golden_ptns, target_mfcc, threshold=threshold, multiproc=multiproc)
    return result

def read_golden_ptns(folderpath):
    """ Read every id pattern in folderpath. If there exists a pickle, use it.

    Parameters
    ----------
    folderpath : (string)
        The folderpath for the golden-pattern wavfiles.

    Return
    ------
    The dictionary of golden patterns.
    """
    try:
        with open('golden_ptns.pickle', 'rb') as pfile:
            return pickle.load(pfile)
    except FileNotFoundError:
        filenames = os.listdir(folderpath)    # list every file in the folderpath
        paths = (join(folderpath, f) for f in filenames if isfile(join(folderpath, f)))
        golden_ptns = dict()
        for idx, path in enumerate(paths):
            (rate, sig) = wav.read(path)
            golden_ptns[filenames[idx]] = mfcc(sig, rate, appendEnergy=False)    # save MFCC feature
        with open("golden_ptns.pickle", 'wb') as pfile:    # save the pickle binary
            pickle.dump(golden_ptns, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    return golden_ptns