import whisper
import sounddevice as sd
import numpy as np
import pyttsx3

model = whisper.load_model("small")

def record_audio(duration=6, fs=16000):
    print("Listening... speak now")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def is_silent(audio, threshold=0.01):
    return np.abs(audio).mean() < threshold

def speech_to_text():
    audio = record_audio()

    if is_silent(audio):
        return ""

    result = model.transcribe(audio, fp16=False, language="english")
    no_speech_prob = result.get("segments", [{}])[0].get("no_speech_prob", 0) if result.get("segments") else 1.0
    if no_speech_prob > 0.6:
        return ""

    text = result["text"].strip()
    print("You said:", text)
    return text

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    print("Starting conversation mode. Say 'stop' to end.")
    while True:
        spoken_text = speech_to_text()

        if spoken_text == "":
            print("Didn't catch anything, listening again...")
            continue

        if "stop" in spoken_text.lower():
            print("Stopping.")
            text_to_speech("Goodbye")
            break

        text_to_speech(spoken_text)