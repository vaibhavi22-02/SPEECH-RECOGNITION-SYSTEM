# SPEECH-RECOGNITION-SYSTEM

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: VAIBHAVI SHINGARE

*INTERN ID*: CT06DA723

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTHOSH

## DESCRIPTION OF TASK
Speech Recognition System using Automatic speech recognition (ASR) in python
The task presented in the given script is focused on automatic speech recognition (ASR), which involves converting spoken language into text. This process is vital in many real-world applications, such as virtual assistants, transcription services, accessibility tools, and more. The script showcases and compares two different approaches to speech-to-text transcription using Python:
Google Speech Recognition API via the speech_recognition library
Wav2Vec2 model from Facebook AI, implemented via Hugging Face Transformers

1. Google Speech Recognition API
The first method uses the speech_recognition library, a popular and easy-to-use Python module that interfaces with several speech-to-text APIs, including Google’s. The transcribe_audio_speech_recognition() function initializes a recognizer, loads the audio file using sr.AudioFile, and records the audio data. It then attempts to transcribe the speech using recognize_google(), which sends the audio to Google’s web service for processing. If the transcription is successful, the resulting text is returned. If the service is unable to understand the audio (UnknownValueError) or cannot reach Google servers (RequestError), appropriate error messages are returned.
This method is simple and accurate for general usage, as it relies on Google’s powerful cloud-based speech recognition engine. However, it requires an internet connection and may have limitations in terms of API quota or data privacy.

2. Wav2Vec2 Transformer Model
The second method, transcribe_audio_wav2vec(), implements a deep learning-based solution using Facebook's Wav2Vec2 model. Wav2Vec2 is a self-supervised model trained on a large corpus of unlabeled audio data and fine-tuned for speech recognition. It is a state-of-the-art approach that operates offline, providing flexibility and data privacy benefits.
This method begins by loading the model and processor using the Hugging Face transformers library. The processor prepares the audio data for the model by tokenizing the waveform into a format suitable for the neural network. Audio is loaded from the given file path using librosa with a fixed sample rate of 16 kHz, which is the standard input format for Wav2Vec2.
After preprocessing, the model performs inference on the input audio tensor. The output logits (raw prediction scores) are passed through a torch.argmax operation to obtain the most likely token IDs. Finally, these tokens are decoded back into human-readable text using the processor’s batch_decode() method.
This deep learning approach offers more control and customization, works offline, and can be fine-tuned for specific domains or accents. However, it is computationally more intensive and requires the necessary machine learning infrastructure.

![Image](https://github.com/user-attachments/assets/c8097d59-13d0-40a6-b400-03afa7c9dcf6)
