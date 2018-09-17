# Telecom Voice Classification

[![pipeline status](https://gitlab.com/GLaDOS1105/telecom-voice-classification/badges/master/pipeline.svg)](https://gitlab.com/GLaDOS1105/telecom-voice-classification/commits/master)
[![coverage report](https://gitlab.com/GLaDOS1105/telecom-voice-classification/badges/master/coverage.svg)](https://gitlab.com/GLaDOS1105/telecom-voice-classification/commits/master)
[![Requirements Status](https://requires.io/github/GLaDOS1105/telecom-voice-classification/requirements.svg?branch=master)](https://requires.io/github/GLaDOS1105/telecom-voice-classification/requirements/?branch=master)

## Contents

* [`televid.py`](televid.py): Provide the basic model for identification of single wave file.
* [`run_through.py`](run_through.py): Only for automatically test every mp3 and wave file in `./test_audio` folder.
* [`run_televid_example.py`](run_televid_example.py): The example for using the Televid model in [`televid.py`](televid.py).
* [`/golden_wav`](golden_wav): The folder contains wave files to generate golden patterns for matching. The [`golden_ptns.pkl`](golden_wav/golden_ptns.pkl) file in it is to speed up the loading (if exists).
* [`/python_speech_features`](python_speech_features): The package for MFCC feature.

## Categories

* In Busy
  * Original Texts
    * 您所撥的電話忙線中，請稍後再撥。 The number you have dial is busy. Please try again later.
  * Keywords
    * 忙線中
* No Response
  * Original Texts
    * 您所撥的電話無人回應。
    * 您所撥的電話無法接聽，請稍後再撥。The number you has dial is not available. Please try again later.
  * Keywords
    * 回應
    * 接聽
* Voice Mail
  * Original Texts
    * 您的電話將轉接到語音信箱，嘟聲後開始計費，如不留言請掛斷。快速留言嘟聲後請按＃字鍵這是09XX-XXXXXX的信箱，嗶聲後請留言。
    * 轉接語音信箱，嘟聲後開始計費，如不留言請掛斷。快速留言嘟聲後請按 * 字鍵。您已進入09XX-XXXXXX的信箱，嗶聲後請留言。
    * 嘟聲後開始計費，如不留言請掛斷。快速留言嘟聲後請按 * 字鍵。您已進入09XX-XXXXXX的信箱，嗶聲後請留言。
    * 您的電話將轉接到語音信箱，嘟聲後開始計費，如不留言請掛斷。快速留言嘟聲後請按一次 * 字鍵。
  * Keywords
    * 語音信箱
    * 嘟聲後

## Spec of `dataset.pkl`

The file is generated by calling `save_mfcc_training_dataset()` of `TestTelevid`
object.

### File Format

Is the pickle, Python object serialization, object. You should use
`pickle.loads()` to open it.

### Content Format

A list contains multiple tuples, and being one data input, each tuple contains
two variables: difference indices dictionary (input) and the final outcome
(desire output). The following is the example structure.

``` python
[
    (
        {'in_busy.wav': 285.38331120979802,
         'no_response_A.wav': 3430.9011939973934,
         'no_response_B.wav': 2380.5107159615013,
         'voice_mail_A_1.wav': 3118.0131102683249,
         'voice_mail_A_2.wav': 2543.842005054099,
         'voice_mail_B.wav': 3217.5192698396595,
         'voice_mail_C.wav': 3176.1896581534188,
         'voice_mail_D_1.wav': 2749.3900560206898,
         'voice_mail_D_2.wav': 2634.9694942389929},
     'inbusy'),
    (
        {'in_busy.wav': 2306.9580628421327,
         'no_response_A.wav':3727.4782066043199,
         'no_response_B.wav': 2987.7147191635795,
         'voice_mail_A_1.wav': 2345.7121554450459,
         'voice_mail_A_2.wav': 2477.9789650889234,
         'voice_mail_B.wav': 328.95530907943493,
         'voice_mail_C.wav': 332.53955702325237,
         'voice_mail_D_1.wav': 2714.4307011467545,
         'voice_mail_D_2.wav': 1705.1556381267567},
     'voicemail')]
```

## Trained Module Input Requirements

The input of the trained module must be a ```dict()``` and then return the
classified result.