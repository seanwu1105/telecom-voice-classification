""" Run the televoice_indentify() through the testing data in `/test_audio`. """

import collections
import csv
from multiprocessing import Process, Queue
import os
from os.path import join, isfile
import time
from televoice_identification import televoice_identify

Result = collections.namedtuple('Result', ['target_fn',
                                           'matched_golden_fn',
                                           'diff_idx',
                                           'successful',
                                           'exe_time'])

def run(folderpath=join("test_audio"), threshold=None, scan_step=1,
        multiproc_cmp=False, nmultiproc_run=8):
    """ Get the comparison result for each testing audio files. Result will be saved in
        `results.csv`.

    Parameters
    ----------
    folderpath : string
        The folderpath of testing audio files. The default is `./test_audio`.
    threshold : float
        The threshold for the least difference to break the comparison.
    scan_step : integer
        The step of scanning on frame of target MFCC pattern.
    multiproc_cmp : boolean
        If `True`, the comparing process will run in multicore of CPU, and vice versa.
    nmultiproc_run : integer
        The # of process in running test. If set `None` or non-positive integer, `run()` will excute
        sequentially. The default is `8`.
    """
    try:
        os.remove("golden_ptns.pickle")    # remove the old golden patterns' pickle
    except OSError:
        pass
    filenames = os.listdir(folderpath)    # list every file in the folderpath
    paths = (join(folderpath, f) for f in filenames if isfile(join(folderpath, f)))
    results = set()

    total_start_time = time.time()
    if nmultiproc_run is not None and nmultiproc_run > 1: # run parallelly
        procs = []
        queue = Queue()
        for idx, path in enumerate(paths):
            if idx != 0 and idx % nmultiproc_run == 0:
                for proc in procs:
                    proc.start()
                for _ in procs:
                    results.add(queue.get())
                procs = []
            procs.append(Process(target=calculate_result,
                                 args=(filenames[idx], path),
                                 kwargs={'threshold': threshold,
                                         'scan_step': scan_step,
                                         'multiproc': multiproc_cmp,
                                         'queue': queue}))
        if procs:
            for proc in procs:
                proc.start()
            for _ in procs:
                results.add(queue.get())
    else: # run sequentially
        for idx, path in enumerate(paths):
            results.add(calculate_result(filenames[idx], path,
                                         threshold=threshold,
                                         scan_step=scan_step,
                                         multiproc=multiproc_cmp))
    total_time = time.time() - total_start_time
    print("------ Total Time Elapse: {} ------".format(total_time))
    # output the result in csv file
    with open("results.csv", 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(('Target', 'Matched', 'Difference', 'Successful', 'Exe Time', total_time))  # field header
        w.writerows((r.target_fn, r.matched_golden_fn, r.diff_idx, r.successful, r.exe_time) for r in results)

def calculate_result(filename, filepath, threshold=None, scan_step=1, multiproc=False, queue=None):
    """ Calculate the result and print on the screen.

    Parameters
    ----------
    filename : string
        The filename to compare whether is successful or not.
    filepath : string
        The filepath to the target audio file.
    threshold : float
        The threshold for the least difference to break the comparison.
    scan_step : integer
        The step of scanning on frame of target MFCC pattern.
    multiproc : boolean
        If `True`, the comparing process will run in multicore of CPU, and vice versa.
    queue : multiprocessing.Queue()
        The `Queue` instance for getting the result by multiprocess `Process()`.
    
    Return
    ------
    result : (namedtuple) A nemedtuple `Result` containing the information of each result.
    """
    start_time = time.time()
    diff_dict = televoice_identify(filepath, threshold=threshold,
                                   scan_step=scan_step, multiproc=multiproc)
    result = Result(filename,
                    min(diff_dict, key=diff_dict.get),
                    min(diff_dict.values()),
                    filename[:2] == min(diff_dict, key=diff_dict.get)[:2],
                    time.time() - start_time)
    print("{:30} {:30}({:8.2f})   {}  {:9.5f}(s)".format(result.target_fn, result.matched_golden_fn,
                                                         result.diff_idx, result.successful,
                                                         result.exe_time))
    if queue is not None:
        queue.put(result)
    return result

if __name__ == '__main__':
    run(multiproc_cmp=True, threshold=1700, scan_step=4)
