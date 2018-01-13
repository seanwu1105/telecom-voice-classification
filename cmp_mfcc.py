""" Compare the MFCC difference between id patterns and target file. """
import numpy as np
import math

def cmp_mfcc(id_ptns, target_ptn):
    # TODO: use the threshold to stop searching instead of comparing whole array.
    """ Compare the MFCC difference. Return the smallest difference number for each id patterns.
        # Parameters
        * `id_ptns` (dict): The id (base) patterns dictionary. Each data is 1-dimension array.
        * `target_ptn` (1darray): The target pattern to be compared.
        """

    id_diff = dict()
    for name, ptn in id_ptns.items():
        window = len(ptn)
        diff = math.inf
        print(range(len(target_ptn) - window))
        for idx in range(len(target_ptn) - window + 1):
            # TODO: check the minimum of target length
            diff = min(sum(np.power(target_ptn[idx:idx + window] - ptn, 2).flat), diff)
        id_diff[name] = diff
    return id_diff
