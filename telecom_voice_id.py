""" The entry points of the identification for telecom mobile voice.

    Author: Sean Wu, Bill Haung
"""

import os
from os.path import join, isfile
import pickle
import sys
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from cmp_mfcc import cmp_mfcc

def main():
    """ The main function for entry point. """

    if len(sys.argv) < 2:    # check the command format
        print("Error: No target file defined. Usage: {} <target_file>".format(sys.argv[0]))
        sys.exit(1)
    else:
        id_patterns = read_id_pattern(join("id_wav"))
        (rate, sig) = wav.read(sys.argv[1])    # read the target wavfile
        target_mfcc = mfcc(sig, rate, appendEnergy=False)
        print(cmp_mfcc(id_patterns, target_mfcc))

def read_id_pattern(folderpath):
    """ Read every id pattern in folderpath. If there exists a pickle, use it. """
    try:
        with open('id_ptn.pickle', 'rb') as pfile:
            id_patterns = pickle.load(pfile)
    except FileNotFoundError:
        filenames = os.listdir(folderpath)    # list every file in the folderpath
        paths = (join(folderpath, f) for f in filenames if isfile(join(folderpath, f)))
        id_patterns = dict()
        for idx, path in enumerate(paths):
            (rate, sig) = wav.read(path)
            id_patterns[filenames[idx]] = mfcc(sig, rate, appendEnergy=False)    # save MFCC feature
        with open("id_ptn.pickle", 'wb') as pfile:
            pickle.dump(id_patterns, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    return id_patterns

if __name__ == '__main__':
    main()