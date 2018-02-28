""" Author: Sean Wu
    NCU CSIE 3B, Taiwan

Run through all of the testing data by calling televid.
"""

import csv
import itertools
import logging
import multiprocessing as mp
import pathlib
import pickle
import platform
import time

import televid

logging.basicConfig(level=logging.INFO)

class TestTelevid(object):
    """ Hold the state of multiple results of `Televid` instance. """

    def __init__(self, folderpath=pathlib.Path('test_audio'), ext=('*.wav', '*.mp3')):
        """ Initialize the folder path and extensions for files to test in
            `TestTelvid().run()`.

        folderpath (str, optional): Defaults to test_audio. The folder path of
            testing audio files.
        ext (tuple, optional): Defaults to ('*.wav', '*.mp3'). The extensions
            (file types) which need to be tested.
        """

        self.total_running_time = None
        self.paths = itertools.chain.from_iterable(folderpath.glob(e) for e in ext)
        self.res = set()
        self.threshold = None
        self.scan_step = None
        self.multiproc_identify = None
        self.nmultiproc_run = None

    def run(self, threshold=None, scan_step=1, multiproc_identify=False,
            nmultiproc_run=8, display_results=True):
        """ Get the comparison result for each testing audio files.

        threshold (float, optional): Defaults to None. The threshold for the
            least difference to stop the comparison procedure.
        scan_step (int, optional): Defaults to 1. The step of scanning on frame
            of target MFCC pattern.
        multiproc_identify (bool, optional): Defaults to False. If set True, the
            comparing process will run in multicore of CPU, and vice versa.
        nmultiproc_run (int, optional): Defaults to 8. The number of process in
            running test. If set None or non-positive integer, `run()` will
            excute sequentially.
        display_results (bool, optional): Defaults to True. If set True, show
            the result in run time.

        Returns:
            set: A set containing all results in testing folder.
        """


        def identify_proc(filepath, mp_queue=None):
            """ Calculate the result by calling the `identify()` of each Televid
                object.

            Args:
                filepath (str): The file path to the target audio file.
                mp_queue (multiprocessing.Queue, optional): Defaults to None.
                    The `Queue` instance for getting the result by multiprocess
                    `Process()`.

            Returns:
                Televid: A Televid instance containing the result after
                    indentified.
            """

            televoice = televid.Televid(filepath)
            televoice.identify(threshold=threshold, scan_step=scan_step,
                               multiproc=multiproc_identify)
            if mp_queue is not None:
                mp_queue.put(televoice)
            return televoice

        start_time = time.time()
        self.threshold = threshold
        self.scan_step = scan_step
        self.multiproc_identify = multiproc_identify
        self.nmultiproc_run = nmultiproc_run

        if nmultiproc_run is None or nmultiproc_run <= 1:
            # Run sequentially
            for path in self.paths:
                output = identify_proc(path)
                self.res.add(output)
                if display_results:
                    self.display(output)
        else:
            # Run parallelly
            mp_queue = mp.Queue()
            procs = []
            for idx, path in enumerate(self.paths):
                if idx != 0 and idx % nmultiproc_run == 0:
                    for proc in procs:
                        proc.start()
                    for _ in procs:
                        output = mp_queue.get()
                        self.res.add(output)
                        if display_results:
                            self.display(output)
                    procs = []
                procs.append(mp.Process(target=identify_proc,
                                        args=(path, mp_queue)))

            # If the number of processes is not divisible by nmultproc_run, get
            # the result of the remaining processes.
            if procs:
                for proc in procs:
                    proc.start()
                for _ in procs:
                    output = mp_queue.get()
                    self.res.add(output)
                    if display_results:
                        self.display(output)

        self.total_running_time = time.time() - start_time
        logging.getLogger(__name__).info("Total time elapse: %f",
                                         self.total_running_time)
        return self.res

    def save_results(self, detailed=True):
        """ Save the results as a readable csv file.

        detailed (bool, optional): Defaults to True. If set True, show all of
            the testing parameters and OS environment in content and name of
            csv file.
        """

        if detailed:
            details = {'total_running_time': self.total_running_time,
                       'platform': platform.system() + ' ' + platform.release(),
                       'threshold': self.threshold, 'scan_step': self.scan_step,
                       'multiproc_identify': self.multiproc_identify,
                       'nmultiproc_run': self.nmultiproc_run}
        else:
            details = dict()
        msg = ['{}={}'.format(key, val) for key, val in details.items()]
        filename = ' '.join(msg)

        with open('{}.csv'.format(filename), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Field header
            writer.writerow(('Name', 'Matched', 'Difference',
                             'Max Result Difference', 'Result Type',
                             'Is Correct', 'Identify Time', ''.join(msg)))
            writer.writerows((r.filepath.name, *r.matched_pattern(True), r.mrd,
                              r.result_type, r.is_correct, r.identify_time)
                             for r in self.res)
        logging.getLogger(__name__).info("Results csv file has generated.")

    def save_mfcc_training_dataset(self):
        """ Generate the dataset of MFCC feature comparison results.

        The dataset can be use to train the machine learning network of
        classifier in different televoice types. The generated file is saved as
        `dataset.pkl`, the pickle binary.
        """

        # Save the dataset as pickle
        with open(pathlib.Path('dataset.pkl'), 'wb') as pfile:
            pickle.dump([(r.diffs, r.result_type) for r in self.res], pfile,
                        protocol=pickle.HIGHEST_PROTOCOL)
        logging.getLogger(__name__).info("dataset.pkl has generated.")

    @staticmethod
    def display(result):
        """ Display the running-time result.

        Args:
            result (Televid): A Televid instance which has already identified.

        Returns:
            Televid: The same object as argument `result`.
        """

        msg_1 = "{:30}{:27}({:8.2f})".format(result.filepath.name,
                                             *result.matched_pattern(True))
        msg_2 = "MRD={:8.2f}{:^17}{:^7}{:9.5f}(s)".format(result.mrd,
                                                          result.result_type,
                                                          result.is_correct,
                                                          result.identify_time)
        print(msg_1, msg_2, sep='\t')
        return result

def main():
    """ The main function. """

    ttvid = TestTelevid()
    ttvid.run(threshold=1500, scan_step=3, multiproc_identify=True)
    ttvid.save_results()

if __name__ == '__main__':
    main()
