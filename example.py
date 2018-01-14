#!/usr/bin/env python

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy

numpy.set_printoptions(threshold=numpy.nan)
(rate,sig) = wav.read("no_response_A (1).wav")
mfcc_feat = mfcc(sig,rate, appendEnergy=False)
#d_mfcc_feat = delta(mfcc_feat, 2)
#fbank_feat = logfbank(sig,rate)
print(mfcc_feat[0:3])
print(len(mfcc_feat))