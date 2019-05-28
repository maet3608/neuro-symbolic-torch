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
"""

import os
import os.path as osp
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk

from tkinter import ttk  # Normal Tkinter.* widgets are not themed!
from ttkthemes import ThemedTk

FONT = 'Calabri', 16


# import tkinter.font as tkFont
# myFont = tkFont.Font(family='Helvetica', size=20)


def center_window(window, size=0.5):
    sw, sh = window.winfo_screenwidth(), window.winfo_screenheight()
    w, h = int(sw * size), int(sh * size)
    x = int((sw - w) / 2)
    y = int((sh - h) / 2)
    window.geometry("{}x{}+{}+{}".format(w, h, x, y))


class App(ttk.Frame):

    def __init__(self, window):
        ttk.Frame.__init__(self, master=window)
        window.title("Neuro-symbolic DR grading")
        window.config(background='white')
        #center_window(window)

        self.load_image()
        self.show_image2()

        btn = ttk.Button(window, text="Quit", command=window.destroy)
        btn.grid(column=1, row=0)

        mic_img = ImageTk.PhotoImage(file='mic.png')
        mic_btn = tk.Button(window, image=mic_img)
        mic_btn.mic_img = mic_img  # keep reference
        mic_btn.grid(column=1, row=1)

    def load_image(self):
        data = np.eye(64) * 150
        # self.image = ImageTk.PhotoImage(Image.open("ball.png"))
        self.image = ImageTk.PhotoImage(image=Image.fromarray(data))


    def show_image(self):
        canvas = tk.Canvas(self, width=200, height=200)
        canvas.create_image(64, 64, anchor="nw", image=self.image)
        canvas.bind('<ButtonPress-1>', self.img_click)
        canvas.grid(column=0, row=0)

    def show_image2(self):
        img = ttk.Label(self.master, image=self.image)
        img.bind('<ButtonPress-1>', self.img_click)
        img.place(x=0, y=0)
        img.grid(column=0, row=0)

    def img_click(self, event):
        print(event)


window = ThemedTk(theme="plastik")  # arc, 	plastik, equilux, aqua, scidgrey
app = App(window)
app.mainloop()

# window = ThemedTk(theme="arc")  # arc, 	plastik, equilux, aqua, scidgrey
# window.title("Demo")
# window.geometry('400x200')
# window.config(background='white')
#
# btn = ttk.Button(window, text="Quit", command=window.destroy)
# btn.grid(column=0, row=0)
#
# lbl = ttk.Label(window, text="Type very long text:", font=FONT)
# lbl.config(background='white')
# lbl.grid(column=0, row=1)
#
# ent = ttk.Entry(window)
# ent.config(font=FONT)
# ent.grid(column=1, row=1)
#
# com = ttk.Combobox(window, values=["January", "February", "March", "April"])
# com.current(1)
# com.config(font=FONT)
# com.grid(column=0, row=2)
#
# window.mainloop()
