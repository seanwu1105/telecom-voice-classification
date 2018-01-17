""" Run the televoice_indentify() through the testing data in `/test_audio`. """

from multiprocessing import Process
import os
from os.path import join, isfile
import time
from televoice_identification import televoice_identify

def main(folderpath=join("test_audio"), threshold=None, multiproc_cmp=True, multiproc_cal=None):
    """ The entry function which reads all testing audio in `folderpath` parameter. """
    filenames = os.listdir(folderpath)    # list every file in the folderpath
    paths = [join(folderpath, f) for f in filenames if isfile(join(folderpath, f))]
    st = time.time()
    for idx, path in enumerate(paths):
        calculate_result(filenames[idx], path, threshold=threshold, multiproc=multiproc_cmp)
    print("------ sequential: {} ------".format(time.time() - st))
    st = time.time()
    procs = []
    for idx, path in enumerate(paths):
        procs.append(Process(target=calculate_result,
                             args=(filenames[idx], path),
                             kwargs={'threshold': threshold, 'multiproc': multiproc_cmp}))
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("------ multicore: {} ------".format(time.time() - st))

def calculate_result(filename, filepath, threshold=None, multiproc=True):
    """ Calculate the result and print on the screen.

    Parameters
    ----------
    filename : (string)
        The filename to compare whether is successful or not.
    filepath : (string)
        The filepath to the target audio file.
    """
    start_time = time.time()
    result = televoice_identify(filepath, threshold=threshold, multiproc=multiproc)
    print("{}:\t\t{}\t\t({})\t{}\t{}(s)".format(filename,
                                                min(result, key=result.get),
                                                min(result.values()),
                                                filename[:2] == min(result, key=result.get)[:2],
                                                time.time() - start_time))

if __name__ == '__main__':
    main(multiproc_cmp=False)
