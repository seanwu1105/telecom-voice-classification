# Spec of `dataset.pkl`

## File Format
Is the pickle, Python object serialization, object. You should use `pickle.loads()` to open it.

## Content Format
A list contains multiple tuples, and being one data input, each tuple contains two variables:
difference indices dictionary (input) and the final outcome (desire output). The following is the
example structure.

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