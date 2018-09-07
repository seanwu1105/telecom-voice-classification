import logging
import unittest

from televid import Televid


class TestClassificationResult(unittest.TestCase):
    def test_inbusy(self):
        expect = 'inbusy'
        classifier = Televid('tests/data/inbusy.mp3',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_noresponse_a(self):
        expect = 'noresponse'
        classifier = Televid('tests/data/noresponse_a.mp3',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_noresponse_b(self):
        expect = 'noresponse'
        classifier = Televid('tests/data/noresponse_b.mp3',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_typical(self):
        expect = 'typical'
        classifier = Televid('tests/data/typical.mp3',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_voicemail_a_1(self):
        expect = 'voicemail'
        classifier = Televid('tests/data/voicemail_a_1.WAV',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_voicemail_a_2(self):
        expect = 'voicemail'
        classifier = Televid('tests/data/voicemail_a_2.WAV',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_voicemail_b(self):
        expect = 'voicemail'
        classifier = Televid('tests/data/voicemail_b.WAV',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_voicemail_c(self):
        expect = 'voicemail'
        classifier = Televid('tests/data/voicemail_c.mp3',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_voicemail_d_1(self):
        expect = 'voicemail'
        classifier = Televid('tests/data/voicemail_d_1.mp3',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

    def test_voicemail_d_2(self):
        expect = 'voicemail'
        classifier = Televid('tests/data/voicemail_d_2.mp3',
                             Televid.load_golden_patterns())
        classifier.identify()
        self.assertEqual(classifier.result_type, expect)

class TestMultiprocessing(unittest.TestCase):
    def test_multiproc_is_televoice(self):
        # inbusy.mp3 is one of the televoice.
        classifier = Televid('tests/data/inbusy.mp3',
                             Televid.load_golden_patterns())
        classifier.identify(multiproc=True)
        self.assertIsNotNone(classifier.result_type)

    def test_multiproc_is_typical(self):
        classifier = Televid('tests/data/typical.mp3',
                             Televid.load_golden_patterns())
        classifier.identify(multiproc=True)
        self.assertIsNotNone(classifier.result_type)
