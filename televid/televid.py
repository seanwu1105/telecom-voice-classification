""" Author: Sean Wu
    NCU CSIE 3B, Taiwan

The indentification for single target audio file. Basically, this file add some
pre-processing and post-processing for the pattern comparison. Following is the
explanation.

Pre-Processing:
    1. Read and get golden patterns
    2. Convert the target waveform file into specific format (also create as
       .tmp file)
    3. Get the MFCC pattern of target file

then feed the MFCC pattern of target file into `pattern_cmp`.

Post-Processing:
    1. Delete .tmp file created by pre-processing step 2.
    2. Return the result of comparison
"""

import io
import math
import multiprocessing as mp
import logging
import pathlib
import pickle
import time

import numpy as np
from scipy.io import wavfile
import ffmpeg

from .python_speech_features import mfcc


logging.basicConfig(level=logging.INFO)


class Televid():
    """ Calculate the difference indices between target audio and each golden
        audio wavfiles.
    """

    def __init__(self, filepath, golden_patterns):
        """ Build the telecomvoice identification object and do the
            pre-processing.

        Pre-Processing:
            1. Read and get golden patterns
            2. Convert the target waveform file into specific format (also
               create as .tmp file)
            3. Get the MFCC pattern of target file

        filepath (str): The path of target file (to be compared).

        Raise:
            FileNotFoundError: Cannot find the target file located in filepath.
        """

        self.filepath = pathlib.Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError('not such file: %s' % str(self.filepath))
        # Contain the golden patterns with its file name as key.
        self.golden_patterns = golden_patterns
        self.diffs = dict()
        self.identify_time = None
        self.threshold = None
        self.scan_step = None

        # Call the ffmpeg to convert (normalize) the input audio into:
        #    sample rate    8000 Hz
        #    bit depth      16
        #    channels       mono (left channel only, since the target channel is
        #                         the left one)

        # Following is the method to call ffmpeg as subprocess.
        # try:
        #     proc = subprocess.run(['ffmpeg', '-y', '-hide_banner',
        #                            '-loglevel', 'panic',
        #                            '-i', str(self.filepath),
        #                            '-af', 'pan=mono|c0=c0',
        #                            '-ar', '8000',
        #                            '-sample_fmt', 's16',
        #                            '-f', 'wav',
        #                            '-'], stdout=subprocess.PIPE)
        # except FileNotFoundError:
        #     logging.getLogger(__name__).error("Require ffmpeg to convert the"
        #                                       "audio in sepcific format.")
        #     sys.exit(2)  # FFmpeg require

        # The following method is to call ffmpeg as pip3 installed python-ffmpeg
        # module.
        stdout, err = (
            ffmpeg
            .input(str(self.filepath))
            .output('-', format='wav', af='pan=mono|c0=c0', ar=8000, sample_fmt='s16')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )

        logging.getLogger(__name__).debug(err)

        # When the output of FFmpeg is sent to stdout, the program does not fill
        # in the RIFF chunk size of the file header. Instead, the four bytes
        # where the chunk size should be are all 0xFF. scipy.io.wavfile.read()
        # expects that value to be correct, so it thinks the length of the chunk
        # is 0xFFFFFFFF bytes. Hence, we need to patch the RIFF chunk size
        # manually before the data is passed to wavfile.read() via an
        # io.BytesIO() object.

        # This is the size of the entire file in bytes minus 8 bytes for the two
        # fields not included in this count: ChunkID and ChunkSize.
        riff_chunk_size = len(stdout) - 8
        quotient = riff_chunk_size

        # Break up the chunk size into four bytes, held in b.
        binarray = list()
        for _ in range(4):
            quotient, remainder = divmod(quotient, 256)  # every 8 bits
            binarray.append(remainder)

        # Replace bytes 4:8 in stdout with the actual size of the RIFF
        # chunk.
        riff = stdout[:4] + bytes(binarray) + stdout[8:]

        # Read the target wave file.
        rate, signal = wavfile.read(io.BytesIO(riff))

        # Get the MFCC feature of target wavfile.
        self.target_mfcc = mfcc(signal, rate, appendEnergy=False)

    def identify(self, threshold=None, scan_step=1, multiproc=False):
        """ Compare the MFCC patterns differences. Return a dict containing all
            differences.

        threshold (int, optional): Defaults to None. The threshold for the
            least difference to stop the comparison.
        scan_step (int, optional): Defaults to 1. The step of scanning on
            frame of target MFCC pattern.
        multiproc (bool, optional): Defaults to False. Enable the
            multiprocessing for each golden patterns comparison.

        Returns:
            dict: A dictionary of differences between each golden pattern.
        """

        start_time = time.time()
        self.threshold = threshold
        self.scan_step = scan_step

        # The stop flag is to signal all the cmp_proc to stop since the result
        # of one of them is smaller than the threshold. This is used in both
        # sequential and parallel comparisons because it's shared memory object.
        stop_flag = mp.Value('H', 0)

        if not multiproc:
            # Sequential comparison
            for name, ptn in self.golden_patterns.items():
                self.diffs.update(self.cmp_proc(name, ptn, stop_flag))
        else:
            # Multiprocessing parallel comparison
            # The queue for outputs of multiprocessing
            queue = mp.Queue()
            # The flag to stop the process from running if set 1
            procs = [mp.Process(target=self.cmp_proc,
                                args=(*i,
                                      stop_flag,
                                      queue))
                     for i in self.golden_patterns.items()]
            for proc in procs:
                proc.start()
            for _ in procs:
                self.diffs.update(queue.get())
        self.identify_time = time.time() - start_time
        return self.diffs

    def cmp_proc(self, name, golden_pattern, stop_flag, mp_queue=None):
        """ The procedure for one golden pattern.

        Args:
            name (str): The name of the golden pattern.
            golden_pattern (numpy.array): The MFCC feature of the golden
                pattern.
            stop_flag (multiprocessing.Value): If set nonzero, this function
                will be stopped for reaching the condition of `threshold`.
            mp_queue (multiprocessing.Queue, optional): Defaults to None.
                The `Queue` instance for getting the result (diff) by
                multiprocessing `Process()`.

        Returns:
            dict: A dictionary contains only one item which key is the name
                and data is the difference value.
        """

        window = len(golden_pattern)
        diff = math.inf
        if len(self.target_mfcc) >= window:
            for i in range(0, len(self.target_mfcc) - window + 1, self.scan_step):
                if stop_flag.value != 0:
                    diff = math.inf
                    break
                diff_arr = self.target_mfcc[i:i + window] - golden_pattern
                diff = min(sum(np.power(diff_arr, 2).flat), diff)
                if self.threshold and diff / window < self.threshold:
                    stop_flag.value = 1
                    break
        else:
            logging.getLogger(__name__).warning("Ignore the comparison of"
                                                "%s since it's shorter than"
                                                "target MFCC.")
        res = {name: diff / window}
        if mp_queue is not None:
            mp_queue.put(res)
        return res

    def matched_pattern(self, diff_value=False):
        """ Get which golden pattern is the matched one.

        diff_value (bool, optional): Defaults to False. Get the difference
            value if set True.

        Returns:
            str: The name of matched pattern. If `diff_value` is set True, the
                rtype will be a tuple as following:
                (matched_pattern_name, matched_pattern_value).
        """

        if diff_value:
            return (min(self.diffs, key=self.diffs.get).lower(),
                    min(self.diffs.values()))
        return min(self.diffs, key=self.diffs.get).lower()

    @property
    def mrd(self):
        """ Get the maximum difference among difference indice. """
        return max(self.diffs.values()) - min(self.diffs.values())

    @property
    def result_type(self):
        """ Check the type of result, which returns the full lowercase string.
            The default typical detect conditions is set by trail and error in
            following settings:
            (threshold=1500, scan_step=3) or (threshold=None, scan_step=1)

        TODO: This can be optimized by machine learning classifier.
        """
        if self.mrd < 2000 and self.matched_pattern(True)[1] > 2000:
            return 'typical'
        return ''.join(self.matched_pattern(True)[0].split('_')[:2]).lower()

    @property
    def is_correct(self):
        """ Check the result is correct or not. This is only the comparison
            between target filename and the `.result_type`.
        """
        return self.filepath.name[:2] == self.result_type[:2]

    @staticmethod
    def load_golden_patterns(folderpath='wav'):
        """ Load every wavfile in folderpath and generate its MFCC feature.

            If there exists a pickle, load it instead. Returns a dict()
            containing MFCC features with its file name as key.

        folderpath (str, optional): Defaults to 'wav'. The relative folder
            path (relative to this script) of the golden wavfiles.

        Returns:
            dict: Contains MFCC features with its file name as key.
        """

        folderpath = pathlib.Path(__file__).parent.joinpath(folderpath)

        # Keep trying to open the pickle file if an error occurs.
        while True:
            try:
                with folderpath.joinpath('golden_ptns.pkl').open('rb') as pfile:
                    golden_patterns = pickle.load(pfile)
                return golden_patterns
            except FileNotFoundError:
                # The pickle file does not exist.
                # Get MFCC feature from golden wavfiles.
                golden_patterns = dict()
                for fpath in folderpath.glob('*.wav'):
                    (rate, sig) = wavfile.read(fpath)
                    golden_patterns[fpath.stem] = mfcc(sig, rate,
                                                       appendEnergy=False)
                # Save the pickle.
                with folderpath.joinpath('golden_ptns.pkl').open('wb') as pfile:
                    pickle.dump(golden_patterns, pfile,
                                protocol=pickle.HIGHEST_PROTOCOL)
                return golden_patterns
            except EOFError:
                # The pickle file created but binary content haven't been
                # written in.
                logging.getLogger(__name__).warning("Load golden_ptns.pkl"
                                                    "but is empty, retrying..")
                continue
            except pickle.UnpicklingError as err:
                logging.getLogger(__name__).warning("Load golden_ptns.pkl"
                                                    "but %s, retrying..", err)
                continue
