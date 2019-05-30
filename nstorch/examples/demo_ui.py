"""
icons:
https://www.materialui.co/icon/mic


TODO;
- answer_question
  - count: say and print number of objs
  - show: segment obj and overlay on image
  - grade: say and print grade
  - what: take mouse position, do all segementation, find hit, print and say obj
"""

import os
import os.path as osp
import pythoncom
import tkinter as tk
import skimage.transform as skt
import speech_recognition as sr

from tkinter import N, E, W, S
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from fundus_generator import gen_images
from threading import Thread
from win32com.client import Dispatch
from train_grading import create_model, predict_one, IH, IW

from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk

FONT = 'Calabri', 16


class Speak(Thread):
    """Text to speech in background thread"""

    def __init__(self, text):
        Thread.__init__(self)
        self.text = text

    def run(self):
        pythoncom.CoInitialize()
        dispatch = Dispatch("SAPI.SpVoice")
        dispatch.Speak(self.text)


def say(text):
    """Speak the given text in background thread"""
    Speak(text).start()


def set_text(box, text):
    start = 0 if isinstance(box, ttk.Entry) else '1.0'
    box.delete(start, tk.END)
    box.insert(start, text)


def speech2text(app):
    app.btn_mic.config(image=app.img_mic_on)
    with sr.Microphone(device_index=app.device_index) as source:
        try:
            set_text(app.txt_out, 'listening...')
            audio = app.recognizer.listen(source)
            app.recognized = app.recognizer.recognize_google(audio)
            set_text(app.ent_cmnd, app.recognized)
            set_text(app.txt_out, '')
            app.execute()
        except sr.UnknownValueError:
            set_text(app.txt_out, 'Could not recognize audio!')
        except sr.RequestError as e:
            set_text(app.txt_out, 'Could not connect: {0}'.format(e))
    app.btn_mic.config(image=app.img_mic_off)


class App(ttk.Frame):
    """The main Application"""

    def __init__(self, window):
        ttk.Frame.__init__(self, master=window)
        window.title("Neuro-symbolic DR grading")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        # window.config(background='white')

        # icons
        self.img_run = ImageTk.PhotoImage(file='run.gif')
        self.img_mic_off = ImageTk.PhotoImage(file='mic_off.gif')
        self.img_mic_on = ImageTk.PhotoImage(file='mic_on.gif')

        self.recognizer = sr.Recognizer()
        self.recognized = ''  # recognized text

        self.model = create_model()
        self.model.load_weights('best_weights.pt')


        conf = {'samples': 10,
                'pathologies': {'ha': [0, 3], 'ex': [0, 4], 'ma': [0, 10]}}
        self.images = list(gen_images(conf, IH, IW))
        self.iidx = 0

        self.scale = 8  # image scale factor
        self.load_image()

        self.img_panel = self.image_panel()
        self.img_panel.grid(column=0, row=0, columnspan=2, rowspan=2,
                            sticky=W + N, padx=5, pady=5)

        btn_prev = ttk.Button(window, text="<<", command=self.prev_img)
        btn_prev.grid(column=0, row=2, sticky=W + N, padx=5, pady=5)

        btn_next = ttk.Button(window, text=">>", command=self.next_img)
        btn_next.grid(column=1, row=2, sticky=E + N, padx=5, pady=5)

        self.ent_cmnd = ttk.Entry(window, width=36)
        self.ent_cmnd.config(font=FONT)
        self.ent_cmnd.grid(column=2, row=0, sticky=W + E + N + S, padx=5,
                           pady=5)
        set_text(self.ent_cmnd, 'What grade is this?')

        btn_run = tk.Button(window, image=self.img_run, borderwidth=0,
                            command=self.execute)
        btn_run.grid(column=3, row=0, sticky=E + N, padx=5, pady=5)

        btn_mic = tk.Button(window, image=self.img_mic_off, borderwidth=0,
                            command=self.listen_mic)
        btn_mic.on = False
        btn_mic.grid(column=4, row=0, sticky=E + N, padx=5, pady=5)
        self.btn_mic = btn_mic

        self.txt_out = ScrolledText(window, width=36, height=19)
        self.txt_out.config(font=FONT)
        self.txt_out.grid(column=2, row=1, columnspan=3, rowspan=1,
                          sticky=W + E + N + S, padx=5, pady=5)

        self.mics = sr.Microphone().list_microphone_names()
        self.device_index = 0
        self.com_mic = ttk.Combobox(window, values=self.mics)
        self.com_mic.bind('<<ComboboxSelected>>', self.select_mic)
        self.com_mic.current(self.device_index)
        self.com_mic.grid(column=2, row=2, columnspan=3,
                          sticky=W + E, padx=5, pady=5)

        # self.window.bind("<Key>", self.key_pressed)

    def contains(self, *words):
        """Returns true if any of the words is in the recognized text"""
        return any(w for w in words if w in app.ent_cmnd.get())

    def translate(self):
        """Translate text into functional program"""
        c = self.contains
        obj = None
        if c('haemorrhage', 'haemorhage', 'memori', 'hemorr'):
            obj = 'ha'
        if c('exudate', 'exit'):
            obj = 'ex'
        if c('micro'):
            obj = 'ma'
        if c('optic disc', 'disc'):
            obj = 'od'
        if c('fovea'):
            obj = 'fo'
        if c('fundus'):
            obj = 'fu'

        if c('show') and obj:
            return 'segment_%s(x)' % obj

        if c('count', 'how many') and obj:
            fn = 'cnt_%s(seg_%s(x))' % (obj, obj)
            answer = 'There are '
            return fn, answer

        return None

    def execute(self):
        """Execute action specified in command field"""
        c = self.contains
        if c('next', 'forward') and c('image'):
            say("okay next image")
            self.next_img()

        elif c('previous', 'back') and c('image'):
            say("okay previous image")
            self.prev_img()

        elif c('end', 'quit', 'finish') and c('program', 'application', 'demo'):
            say("As you wish my master")
            self.master.destroy()
        else:
            fn, ans = self.translate()
            if not fn:
                say("I don't understand")
                return
            y = predict_one(self.model, fn, self.imgarr)
            print('predict_one', fn, y)
            y = str(round(y.item()))
            text = ans + str(y)
            set_text(app.txt_out, text)
            say(text)

    def next_img(self):
        self.iidx = min(self.iidx + 1, len(self.images) - 1)
        self.load_image()
        self.img_panel.config(image=self.image)

    def prev_img(self):
        self.iidx = max(self.iidx - 1, 0)
        self.load_image()
        self.img_panel.config(image=self.image)

    def listen_mic(self):
        thread = Thread(target=speech2text, args=(self,))
        thread.start()

    def select_mic(self, event):
        name = self.com_mic.get()
        self.device_index = self.mics.index(name)
        with sr.Microphone(device_index=self.device_index) as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def load_image(self):
        self.imgarr = self.images[self.iidx]
        imgscaled = skt.rescale(self.imgarr, scale=self.scale, order=0,
                             multichannel=True,
                             anti_aliasing=None, anti_aliasing_sigma=None,
                             preserve_range=True).astype('uint8')
        pilimg = Image.fromarray(imgscaled)
        self.image = ImageTk.PhotoImage(image=pilimg)

    def image_panel(self):
        panel = ttk.Label(self.master, image=self.image)
        panel.bind('<ButtonPress-1>', self.img_click)
        panel.place(x=0, y=0)
        return panel

    def img_click(self, event):
        c = event.x // self.scale
        r = event.y // self.scale
        print(r, c)
        self.images[self.iidx][r, c, :] = (0, 255, 0)
        self.load_image()
        self.img_panel.config(image=self.image)

    def key_pressed(self, event):
        print("key pressed:", event)
        if event.keycode == 39:  # cursor left
            self.next_img()
        if event.keycode == 37:  # cursor right
            self.prev_img()


window = ThemedTk(theme='plastik')  # arc,plastik,equilux,aqua,scidgrey
app = App(window)
app.mainloop()
