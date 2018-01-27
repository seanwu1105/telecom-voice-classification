""" Run the televoice_indentify() through the testing data in `/test_audio`. """

import collections
import csv
from multiprocessing import Process, Queue
import os
from os.path import join, isfile, exists, basename
import time
import platform
from televoice_identification import televoice_identify

Result = collections.namedtuple('Result', ['target_fn',
                                           'matched_golden_fn',
                                           'diff_idx',
                                           'successful',
                                           'max_result_diff',
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
    if not exists(join("temp")):
        os.makedirs(join("temp"))
    if not exists(join("test_audio")):
        os.makedirs(join("test_audio"))
    try:
        os.remove(join("temp", "golden_ptns.pickle"))    # remove the old golden patterns' pickle
    except OSError:
        print("golden_ptns.pickle not exists, and it's ok.")
    paths = (join(folderpath, f) for f
             in os.listdir(folderpath)
             if isfile(join(folderpath, f)) and f.lower().endswith(('.mp3', '.wav')))
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
                                 args=(basename(path), path),
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
        for path in paths:
            results.add(calculate_result(basename(path), path,
                                         threshold=threshold,
                                         scan_step=scan_step,
                                         multiproc=multiproc_cmp))
    total_time = time.time() - total_start_time
    print("------ Total Time Elapse: {} ------".format(total_time))
    # output the result in csv file
    parameter_msg = ("platform={} {} threshold={} scan_step={} multiproc_cmp={} "
                     "nmultiproc_run={}").format(platform.system(), platform.release(), threshold,
                                                 scan_step, multiproc_cmp, nmultiproc_run)
    with open("{}.csv".format(parameter_msg), 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(('Target', 'Matched', 'Difference', 'Successful', 'Max Result Difference',
                    'Exe Time', total_time, parameter_msg))  # field header
        w.writerows((r.target_fn, r.matched_golden_fn, r.diff_idx, r.successful, r.max_result_diff,
                     r.exe_time) for r in results)

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
                    check_result(filename, diff_dict),
                    max_result_diff(diff_dict),
                    time.time() - start_time)
    print("{:30} {:27}({:8.2f}) {:^7} MRD={:8.2f}  {:9.5f}(s)".format(result.target_fn,
                                                                      result.matched_golden_fn,
                                                                      result.diff_idx,
                                                                      result.successful,
                                                                      result.max_result_diff,
                                                                      result.exe_time))
    if queue is not None:
        queue.put(result)
    return result

def check_result(filename, diff_dict):
    """ Return the result is correct, failed or typical televoice. """
    if max_result_diff(diff_dict) < 2000 and min(diff_dict.values()) > 2000: # XXX: the typical detect conditions
        return 'Typical'
    else:
        return str(filename[:2] == min(diff_dict, key=diff_dict.get)[:2])

def max_result_diff(diff_dict):
    """ Return the maximum difference index of result dictionary. """
    return max(diff_dict.values()) - min(diff_dict.values())

if __name__ == '__main__':
    run(scan_step=3, threshold=1500)
