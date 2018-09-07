""" Author: Sean Wu
    NCU CSIE 3B, Taiwan

Example for identification of single televid.
"""

import logging

from televid import televid

logging.basicConfig(level=logging.INFO)


def main():
    """ The main function which is required for calling telvid.identify() since
    the multiprocessing requires importing main to be frozen to produce an
    executable. Some extra info can be found here: https://github.com/nipy/dipy/issues/399/
    """

    # Load golden patterns first.
    golden_patterns = televid.Televid.load_golden_patterns()

    # Set the target file path.
    filepath = 'tests/data/no_response_B (1).mp3'

    televoice = televid.Televid(filepath, golden_patterns)

    # NOTE: If running on Windows, multiproc may increase running time
    # drastically.
    televoice.identify(threshold=1500, scan_step=4, multiproc=True)

    logging.getLogger(__name__).info('Result: %s', televoice.matched_pattern())
    logging.getLogger(__name__).info('Total time elapse: %f', televoice.identify_time)


if __name__ == '__main__':
    main()