""" Compare the MFCC difference between id patterns and target file. """

import math
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
    for name, ptn in id_ptns.items():
        window = len(ptn)
        diff = math.inf
        # if the length of target pattern is smaller than the id pattern, the difference index will
        #   be infinite.
        if len(target_ptn) >= window:
            for idx in range(len(target_ptn) - window + 1):
                diff = min(sum(np.power(target_ptn[idx:idx + window] - ptn, 2).flat), diff)
        id_diff[name] = diff / window    # normalized the difference index
    return id_diff
