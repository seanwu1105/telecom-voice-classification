""" Run the televoice_indentify() through the testing data in `/test_audio`. """

import os
from os.path import join, isfile
from televoice_identification import televoice_identify

def main(folderpath=join("test_audio")):
    """ The entry function which reads all testing audio in `folderpath` parameter."""
    filenames = os.listdir(folderpath)    # list every file in the folderpath
    paths = (join(folderpath, f) for f in filenames if isfile(join(folderpath, f)))
    for idx, path in enumerate(paths):
        calculate_result(filenames[idx], path)

def calculate_result(filename, filepath):
    """ Calculate the result and print on the screen.

    Parameters
    ----------
    filename : (string)
        The filename to compare whether is successful or not.
    filepath : (string)
        The filepath to the target audio file.
    """
    result = televoice_identify(filepath)
    print("{}:\t\t{}\t\t({})\t{}".format(filename,
                                         min(result, key=result.get),
                                         min(result.values()),
                                         filename[:2] == min(result, key=result.get)[:2]))

if __name__ == '__main__':
    main()
