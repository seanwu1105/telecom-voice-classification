""" Compare the MFCC difference between id patterns and target file. For speeding up, using the
    multiprocessing to share the loading on multicore.

    Author: Sean Wu, Bill Haung
    NCU CSIE 3B, Taiwan
"""

import math
from multiprocessing import Process, Queue
import numpy as np

def cmp_mfcc(id_ptns, target_ptn):
    """ Compare the MFCC difference. Return the smallest difference number for each id patterns.

        Parameters
        ----------
        id_ptns : dict
            The id (base) patterns dictionary. Each data is 1-dimension array.
        target_ptn : 1d-array
            The target pattern to be compared.
        """

    id_diff = dict()    # the difference index dictionary for each id pattern
    queue = Queue()    # the queue for outputs of multiprocessing
    procs = [Process(target=cmp_proc, args=(item, target_ptn, queue)) for item in id_ptns.items()]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
    while not queue.empty():
        id_diff.update(queue.get())
    return id_diff

def cmp_proc(id_item, target_ptn, queue):
    """ The comparing procedure for multiprocessing. """

    window = len(id_item[1])    # the pattern of the id
    diff = math.inf
    # if the length of target pattern is smaller than the id pattern, the difference index will
    #   be infinite.
    if len(target_ptn) >= window:
        for idx in range(len(target_ptn) - window + 1):
            diff = min(sum(np.power(target_ptn[idx:idx + window] - id_item[1], 2).flat), diff)
    queue.put({id_item[0]: diff / window})
