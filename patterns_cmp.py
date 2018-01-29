""" Compare the MFCC difference between id patterns and target file. For speeding up, using the
    multiprocessing to share the loading on multicore. However, this may not always result
    positively.

    Author: Sean Wu, Bill Haung
    NCU CSIE 3B, Taiwan
"""

import math
from multiprocessing import Process, Queue, Value
import numpy as np

def ptns_cmp(golden_ptns, target_ptn, threshold=None, scan_step=1, multiproc=False):
    """ Compare the MFCC patterns difference. Return the smallest difference number for each id
        patterns.

        Parameters
        ----------
        golden_ptns : dict
            The golden patterns dictionary. Each data is 1-dimension array.
        target_ptn : 1d-array
            The target pattern to be compared.
        threshold : float
            The threshold for the least difference to break the comparison.
        scan_step : integer
            The step of scanning on frame of target MFCC pattern.
        multiproc : boolean
            Enable the multiprocessing for each id patterns comparison.

        Return
        ------
        diff_indice : (dict) A dictionary of differences between each golden pattern.
        """

    diff_indice = dict()    # the difference index dictionary for each golden pattern
    if not multiproc:    # sequential comparison
        for name, ptn in golden_ptns.items():
            window = len(ptn)
            diff = math.inf
            # if the length of target pattern is smaller than the golden pattern, the difference
            #     index will be infinite.
            if len(target_ptn) >= window:
                for idx in range(0, len(target_ptn) - window + 1, scan_step):
                    diff = min(sum(np.power(target_ptn[idx:idx + window] - ptn, 2).flat), diff)
                    if threshold and diff / window < threshold:
                        diff_indice[name] = diff / window
                        return diff_indice
            diff_indice[name] = diff / window # save the difference index
    else:    # multicore parallel comparison
        queue = Queue()    # the queue for outputs of multiprocessing
        stop_flag = Value('H', 0)    # the flag to stop the process from running if set 1
        procs = [Process(target=cmp_proc, args=(i,
                                                target_ptn,
                                                queue,
                                                stop_flag,
                                                threshold,
                                                scan_step)) for i in golden_ptns.items()]
        for proc in procs:
            proc.start()
        for _ in procs:
            diff_indice.update(queue.get())
    return diff_indice

def cmp_proc(golden_item, target_ptn, queue, stop_flag, threshold, scan_step):
    """ The comparing procedure for multiprocessing.

    Parameters
        ----------
        golden_item : dict
            The golden patterns dictionary. Each data is 1-dimension array.
        target_ptn : 1d-array
            The target pattern to be compared.
        queue : multiprocessing.Queue()
            The `Queue` instance for getting the result (diff) by multiprocess `Process()`.
        stop_flag: multiprocessing.Value()
            If set nonzero, this function will stop.
        threshold : float
            The threshold for the least difference to break the comparison.
        scan_step : integer
            The step of scanning on frame of target MFCC pattern.
    """

    window = len(golden_item[1])    # the pattern of the golden element in dictionary
    diff = math.inf
    # if the length of target pattern is smaller than the golden pattern, the difference index will
    #     be infinite.
    if len(target_ptn) >= window:
        for idx in range(0, len(target_ptn) - window + 1, scan_step):
            if stop_flag.value == 1:
                break
            diff = min(sum(np.power(target_ptn[idx:idx + window] - golden_item[1], 2).flat), diff)
            if threshold and diff / window < threshold:
                stop_flag.value = 1
    queue.put({golden_item[0]: diff / window})
