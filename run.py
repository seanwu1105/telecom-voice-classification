""" Author: Sean Wu
    NCU CSIE 3B, Taiwan
Run the televoice_indentify() through all of testing data in `/test_audio`. """

import csv
from multiprocessing import Process, Queue
import os
import time
import logging
import pickle
import platform
from televoice_identification import televoice_identify


logging.basicConfig(level=logging.INFO)


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
    logger = logging.getLogger(__name__)
    if not os.path.exists(os.path.join("temp")):
        os.makedirs(os.path.join("temp"))
    if not os.path.exists(os.path.join("test_audio")):
        os.makedirs(os.path.join("test_audio"))
    try:
        # remove the old pickle file
        os.remove(os.path.join("temp", "golden_ptns.pkl"))
    except OSError:
        logger.info("golden_ptns.pkl not exists, and it's ok.")
    paths = (os.path.join(folderpath, f) for f
             in os.listdir(folderpath)
             if f.lower().endswith(('.mp3', '.wav')))
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
    logger.info("Total Time Elapse: %f", total_time)

    # output the result in csv file
    parameters_msg = ("platform={} {} threshold={} scan_step={} multiproc_cmp={} "
                      "nmultiproc_run={}").format(platform.system(), platform.release(), threshold,
                                                  scan_step, multiproc_cmp, nmultiproc_run)
    save_results_csv(results, total_time, parameters_msg)

    return results


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
    print("{:30}{:27}({:8.2f})\tMRD={:8.2f}{:^17}{:^7}{:9.5f}(s)".format(result.filename,
                                                                         *result.matched_golden_ptn,
                                                                         result.mrd,
                                                                         result.result_type,
                                                                         result.is_correct,
                                                                         result.exe_time))
    if queue is not None:
        queue.put(result)
    return result


def save_results_csv(results, total_time, filename='result.csv'):
    """ Save the results as csv readable file. """
    with open("{}.csv".format(filename), 'w', newline='') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(('Target', 'Matched', 'Difference', 'Max Result Difference', 'Result Type',
                    'Is Correct', 'Exe Time', total_time, filename))  # field header
        w.writerows((r.filename, r.matched_golden_ptn[0], r.matched_golden_ptn[1], r.mrd,
                     r.result_type, r.is_correct, r.exe_time) for r in results)


def generate_mfcc_dataset(results):
    """ Generate the dataset of MFCC feature comparison results from test_audio. The dataset can be
    use to train the machine learning network of classifier in different televoice types. The
    generated file is saved as `dataset.pkl`, the pickle binary.
    """
    with open(os.path.join("dataset.pkl"), 'wb') as pfile:  # save the dataset as pickle
        pickle.dump([(r.diff_indice, r.result_type) for r in results], pfile,
                    protocol=pickle.HIGHEST_PROTOCOL)
    logging.getLogger(__name__).info("dataset.pkl has generated.")


if __name__ == '__main__':
    run(threshold=1500, scan_step=3, multiproc_cmp=True)
