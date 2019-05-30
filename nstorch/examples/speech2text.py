"""
Text to speech in background thread.
"""
import pythoncom
from threading import Thread
from win32com.client import Dispatch


class Speak(Thread):
    def __init__(self, text):
        Thread.__init__(self)
        self.text = text

    def run(self):
        pythoncom.CoInitialize()
        dispatch = Dispatch("SAPI.SpVoice")
        dispatch.Speak(self.text)


if __name__ == '__main__':
    speak = Speak('Yes my master')
    speak.start()
