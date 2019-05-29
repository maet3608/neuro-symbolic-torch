"""
TODO:

layout:
https://effbot.org/tkinterbook/grid.htm
https://www.delftstack.com/tutorial/tkinter-tutorial/tkinter-geometry
-managers/
http://zetcode.com/tkinter/layout/

fonts:
https://stackoverflow.com/questions/20588417/how-to-change-font-and
-size-of-buttons-and-frame-in-tkinter-using-python

icons:
https://www.materialui.co/icon/mic

background listening:
https://github.com/Uberi/speech_recognition/blob/master/examples/background_listening.py
"""

import os
import os.path as osp
import tkinter as tk
import skimage.transform as skt
import speech_recognition as sr

from tkinter import N, E, W, S
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from fundus_generator import gen_images

from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk

FONT = 'Calabri', 16


def set_text(box, text):
    start = 0 if isinstance(box, ttk.Entry) else '1.0'
    box.delete(start, tk.END)
    box.insert(start, text)


class App(ttk.Frame):

    def __init__(self, window):
        ttk.Frame.__init__(self, master=window)
        window.title("Neuro-symbolic DR grading")
        # window.config(background='white')

        self.recognizer = sr.Recognizer()

        conf = {'samples': 10,
                'pathologies': {'ha': [0, 3], 'ex': [0, 4], 'ma': [0, 10]}}
        ir, ic = 64, 64
        self.images = list(gen_images(conf, ir, ic))
        self.iidx = 0

        self.scale = 8  # image scale factor
        self.load_image()

        self.img_panel = self.image_panel()
        self.img_panel.grid(column=0, row=0, columnspan=2, rowspan=2,
                            sticky=W + N, padx=5, pady=5)

        btn_prev = ttk.Button(window, text="<<", command=self.prev_img)
        btn_prev.grid(column=0, row=2, sticky=W + E + N, padx=5, pady=5)

        btn_next = ttk.Button(window, text=">>", command=self.next_img)
        btn_next.grid(column=1, row=2, sticky=W + E + N, padx=5, pady=5)

        self.ent_cmnd = ttk.Entry(window, width=36)
        self.ent_cmnd.config(font=FONT)
        self.ent_cmnd.grid(column=2, row=0, sticky=W + E + N + S, padx=5, pady=5)
        set_text(self.ent_cmnd, 'What grade is this?')

        run_img = ImageTk.PhotoImage(file='run.gif')
        btn_run = tk.Button(window, image=run_img, borderwidth=0)
        btn_run.img = run_img  # keep reference
        btn_run.grid(column=3, row=0, sticky=E + N, padx=5, pady=5)

        img_mic_off = ImageTk.PhotoImage(file='mic_off.gif')
        img_mic_on = ImageTk.PhotoImage(file='mic_on.gif')
        btn_mic = tk.Button(window, image=img_mic_off, borderwidth=0,
                            command=self.listen_mic)
        btn_mic.img_mic_off = img_mic_off  # keep reference
        btn_mic.img_mic_on = img_mic_on  # keep reference
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

    def next_img(self):
        self.iidx = min(self.iidx + 1, len(self.images) - 1)
        self.load_image()
        self.img_panel.config(image=self.image)

    def prev_img(self):
        self.iidx = max(self.iidx - 1, 0)
        self.load_image()
        self.img_panel.config(image=self.image)

    def listen_mic(self):
        btn = self.btn_mic
        btn.config(image=btn.img_mic_on)
        self.speech2text()
        btn.config(image=btn.img_mic_off)

    def select_mic(self, event):
        name = self.com_mic.get()
        self.device_index = self.mics.index(name)

    def load_image(self):
        imgarr = self.images[self.iidx]
        imgarr = skt.rescale(imgarr, scale=self.scale, order=0,
                             multichannel=True,
                             anti_aliasing=None, anti_aliasing_sigma=None,
                             preserve_range=True).astype('uint8')
        pilimg = Image.fromarray(imgarr)
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

    def speech2text(self):
        with sr.Microphone(device_index=self.device_index) as source:
            try:
                # self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
                recognized = self.recognizer.recognize_google(audio)
                set_text(self.ent_cmnd, recognized)
            except sr.UnknownValueError:
                set_text(self.txt_out, 'Could not recognize audio!')
            except sr.RequestError as e:
                set_text(self.txt_out, 'Recognition failed: {0}'.format(e))


window = ThemedTk(theme='arc')  # arc,plastik,equilux,aqua,scidgrey
app = App(window)
app.mainloop()
