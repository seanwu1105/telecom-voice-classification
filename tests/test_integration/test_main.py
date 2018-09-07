import unittest

from main import RunTelevid


class TestRunTelevid(unittest.TestCase):
    expects = {
        ('voicemail_c.mp3', 'voice_mail_c', 'voicemail', True),
        ('noresponse_b.mp3', 'no_response_b', 'noresponse', True),
        ('voicemail_d_2.mp3', 'voice_mail_d_2', 'voicemail', True),
        ('voicemail_d_1.mp3', 'voice_mail_d_1', 'voicemail', True),
        ('inbusy.mp3', 'in_busy', 'inbusy', True),
        ('noresponse_a.mp3', 'no_response_a', 'noresponse', True),
        ('typical.mp3', 'voice_mail_d_2', 'typical', True)
    }

    def test_singleproc(self):
        details = RunTelevid('tests/data').run(nmultiproc_run=1,
                                               display_results=False)
        results = {(r.filepath.name, r.matched_pattern(False),
                    r.result_type, r.is_correct) for r in details}
        self.assertEqual(results, self.expects)

    def test_multiproc_run(self):
        details = RunTelevid('tests/data').run(display_results=False)
        results = {(r.filepath.name, r.matched_pattern(False),
                    r.result_type, r.is_correct) for r in details}
        self.assertEqual(results, self.expects)

    def test_multiproc_run_and_identify(self):
        details = RunTelevid('tests/data').run(display_results=False,
                                               multiproc_identify=True)
        results = {(r.filepath.name, r.matched_pattern(False),
                    r.result_type, r.is_correct) for r in details}
        self.assertEqual(results, self.expects)
