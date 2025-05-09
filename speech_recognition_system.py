import speech_recognition as sr
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_audio_speech_recognition(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition"

def transcribe_audio_wav2vec(audio_path):
    # Load model & processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load audio
    audio_input, rate = librosa.load(audio_path, sr=16000)
    inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)

    # Inference
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

audio_path = "sample_audio.wav"
print(transcribe_audio_speech_recognition(audio_path))
