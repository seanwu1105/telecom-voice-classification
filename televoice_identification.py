""" The indentification for single target audio file. """

import os
from os.path import join, isfile
import pickle
import sys
import subprocess
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from patterns_cmp import ptns_cmp

def televoice_identify(filepath, threshold=None, scan_step=1, multiproc=False):
    """ Calculate the difference indices between target audio and each golden audio wavfiles.

    Parameters
    ----------
    filepath : string
        The path of target file (to be compared).
    threshold : float
        The threshold for the least difference to break the comparison.
    scan_step : integer
        The step of scanning on frame of target MFCC pattern.
    multiproc : boolean
        If `True`, the comparing process will run in multicore of CPU, and vice versa.

    Return
    ------
    The dictionary containing the difference indices between target audio and each golden audio
      wavfiles.
    """

    golden_ptns = read_golden_ptns(join("golden_wav")) # load golden wavfiles

    # the filepath for converted wavfile by ffmpeg
    tmp_filepath = join("temp", os.path.basename(filepath) + ".tmp")

    # Call the ffmpeg to convert (normalize) the input audio into:
    #    sample rate    8000 Hz
    #    bit depth      16
    #    channels       mono (left channel only, since the target channel is the left one)
    try:
        subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'panic', '-i', filepath,
                        '-af', 'pan=mono|c0=c0', '-ar', '8000', '-sample_fmt', 's16', '-f', 'wav',
                        tmp_filepath])
    except FileNotFoundError:
        print("[Error] Require ffmpeg to convert the audio in sepcific format.")
        sys.exit(2)    # ffmpeg require
    (rate, sig) = wav.read(tmp_filepath) # read the target wavfile
    target_mfcc = mfcc(sig, rate, appendEnergy=False)
    result = ptns_cmp(golden_ptns, target_mfcc, threshold=threshold, scan_step=scan_step,
                      multiproc=multiproc)
    try:
        os.remove(tmp_filepath) # remove the tmp file
    except OSError:
        pass
    return result

def read_golden_ptns(folderpath):
    """ Read every id pattern in folderpath. If there exists a pickle, use it.

    Parameters
    ----------
    folderpath : string
        The folderpath for the golden-pattern wavfiles.

    Return
    ------
    The dictionary of golden patterns.
    """
    while True:
        try:
            with open(join("temp", "golden_ptns.pickle"), 'rb') as pfile:
                return pickle.load(pfile)
        except FileNotFoundError:
            filenames = os.listdir(folderpath)    # list every file in the folderpath
            paths = (join(folderpath, f) for f in filenames if isfile(join(folderpath, f)))
            golden_ptns = dict()
            for idx, path in enumerate(paths):
                (rate, sig) = wav.read(path)
                golden_ptns[filenames[idx]] = mfcc(sig, rate, appendEnergy=False)# save MFCC feature
            with open(join("temp", "golden_ptns.pickle"), 'wb') as pfile: # save the pickle binary
                pickle.dump(golden_ptns, pfile, protocol=pickle.HIGHEST_PROTOCOL)
            return golden_ptns
        except EOFError: # the pickle file created but binary content haven't been written in
            continue # retry opening pickle file
