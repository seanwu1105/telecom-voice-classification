""" Author: Sean Wu
    NCU CSIE 3B, Taiwan
Run the televoice_indentify() through all of testing data in `/test_audio`. """

import csv
from multiprocessing import Process, Queue
import os
import time
import platform
from televoice_identification import televoice_identify

def run(folderpath=os.path.join("test_audio"), threshold=None, scan_step=1,
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
    if not os.path.exists(os.path.join("temp")):
        os.makedirs(os.path.join("temp"))
    if not os.path.exists(os.path.join("test_audio")):
        os.makedirs(os.path.join("test_audio"))
    try:
        os.remove(os.path.join("temp", "golden_ptns.pickle")) # remove the old pickle file
    except OSError:
        print("golden_ptns.pickle not exists, and it's ok.")
    paths = (os.path.join(folderpath, f) for f
             in os.listdir(folderpath)
             if (os.path.isfile(os.path.join(folderpath, f)) and
                 f.lower().endswith(('.mp3', '.wav'))))
    results = set()

    total_start_time = time.time()
    if nmultiproc_run is not None and nmultiproc_run > 1:
        # run parallelly
        # the variable "nmultiproc_run" is the number of multiprocessing running parallelly one time
        procs = []
        queue = Queue()
        for idx, path in enumerate(paths):
            if idx != 0 and idx % nmultiproc_run == 0:
                for proc in procs:
                    proc.start()
                for _ in procs:
                    results.add(queue.get())
                procs = []
            procs.append(Process(target=calculate_result, args=(path,),
                                 kwargs={'threshold': threshold, 'scan_step': scan_step,
                                         'multiproc': multiproc_cmp, 'queue': queue}))
        if procs:
            for proc in procs:
                proc.start()
            for _ in procs:
                results.add(queue.get())
    else:
        # run sequentially
        for path in paths:
            results.add(calculate_result(path, threshold, scan_step, multiproc_cmp))

    total_time = time.time() - total_start_time
    print("------ Total Time Elapse: {} ------".format(total_time))
    # output the result in csv file
    parameters_msg = ("platform={} {} threshold={} scan_step={} multiproc_cmp={} "
                      "nmultiproc_run={}").format(platform.system(), platform.release(), threshold,
                                                  scan_step, multiproc_cmp, nmultiproc_run)
    with open("{}.csv".format(parameters_msg), 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(('Target', 'Matched', 'Difference', 'Max Result Difference', 'Result Type',
                    'Exe Time', total_time, parameters_msg))  # field header
        w.writerows((r.filename, r.matched_golden_ptn[0], r.matched_golden_ptn[1], r.mrd,
                     r.result_type, r.exe_time) for r in results)

def calculate_result(filepath, threshold=None, scan_step=1, multiproc=False, queue=None):
    """ Calculate the result and print on the screen.

    Parameters
    ----------
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
    result : A special object contaning both raw and analyzed results.
    """

    result = televoice_identify(filepath, threshold, scan_step, multiproc)
    print("{:30}{:27}({:8.2f})\tMRD={:8.2f}{:^11}{:9.5f}(s)".format(result.filename,
                                                                    *result.matched_golden_ptn,
                                                                    result.mrd,
                                                                    result.result_type,
                                                                    result.exe_time))
    if queue is not None:
        queue.put(result)
    return result

if __name__ == '__main__':
    run(scan_step=3, threshold=1500)
