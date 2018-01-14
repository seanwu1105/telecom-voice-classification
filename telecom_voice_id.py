""" The entry points of the identification for telecom mobile voice.

    Author: Sean Wu, Bill Haung
"""

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
        id_patterns = read_id_pattern()
        (rate, sig) = wav.read(sys.argv[1])    # read the target wavfile
        target_mfcc = mfcc(sig, rate, appendEnergy=False)
        print(cmp_mfcc(id_patterns, target_mfcc))

def read_id_pattern():
    #TODO: do not read the id pattern waveforms; instead, read the numpy array data.
    filenames = ["in_busy_main.wav", "no_response_A_main.wav", "no_response_B_main.wav",
                 "voice_mail_A_1_main.wav", "voice_mail_A_2_main.wav", "voice_mail_B_main.wav",
                 "voice_mail_C_main.wav", "voice_mail_D_1_main.wav", "voice_mail_D_2_main.wav"]
    id_patterns = dict()
    for filename in filenames:
        (rate, sig) = wav.read(filename)
        id_patterns[filename] = mfcc(sig, rate, appendEnergy=False)
    return id_patterns

if __name__ == '__main__':
    main()