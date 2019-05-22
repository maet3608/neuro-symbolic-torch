"""
https://win32com.goermezer.de/microsoft/speech-engine/speech-recognition.html
http://blog.justsophie.com/python-speech-to-text-with-pocketsphinx/
http://code.activestate.com/recipes/576412-speech-recognition-and-voice
-synthesis-in-python-f/

conda install pyaudio
pip install SpeechRecognition
"""


def text2speech(text='What is your wish my master'):
    from win32com.client import Dispatch
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(text)


def list_microphones():
    import speech_recognition as sr
    names = sr.Microphone().list_microphone_names()
    for i, name in enumerate(names):
        print(i, name)


def speech2text():
    import speech_recognition as sr
    recording = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        # recording.adjust_for_ambient_noise(source)
        print("listening...")
        audio = recording.listen(source)
        try:
            recognized = recording.recognize_google(audio)
            print("recognized:", recognized)
            return recognized
        except Exception as e:
            print(e)
            return ''


if __name__ == '__main__':
    list_microphones()
    text2speech("there are 0 balls")
    # speech2text()