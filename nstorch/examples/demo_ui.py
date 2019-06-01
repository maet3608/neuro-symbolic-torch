"""
Demo application for neuro-symbolic computing.
Analyzes synthetic fundus images with simulated diabetic retinopathy
and performs visual question answering with speech recognition and
synthesis.
"""
import time
import pythoncom

import numpy as np
import tkinter as tk
import skimage.transform as skt
import speech_recognition as sr

from random import choice
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
    speak = Speak(text)
    speak.start()
    return speak


def load_outofset():
    for filename in ['mars.jpg', 'fundus.jpg']:
        img = Image.open(filename)
        img.thumbnail((IW, IH), Image.ANTIALIAS)
        yield np.asarray(img)


def speech2text(app):
    app.btn_mic.config(image=app.img_mic_on)
    with sr.Microphone(device_index=app.device_index) as source:
        try:
            app.console('listening...')
            audio = app.recognizer.listen(source)
            app.recognized = app.recognizer.recognize_google(audio)
            app.cmdline(app.recognized)
            app.console('')
            app.execute()
        except sr.UnknownValueError:
            app.console('Could not recognize audio!')
        except sr.RequestError as e:
            app.console('Could not connect: {0}'.format(e))
        except Exception as e:
            say('I do not understand')
            app.console('Error: {}'.format(e))
    app.btn_mic.config(image=app.img_mic_off)


def center_window(window):
    w = window.winfo_screenwidth()
    h = window.winfo_screenheight()
    size = tuple(int(pos) for pos in window.geometry().split('+')[0].split('x'))
    x = w / 2 - size[0] / 2
    y = h / 2 - size[1] / 2
    window.geometry("%dx%d+%d+%d" % (size + (x, y)))


class App(ttk.Frame):
    """The main Application"""

    def __init__(self, window):
        ttk.Frame.__init__(self, master=window)
        window.title("Neuro-symbolic DR grading")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)
        window.config(background='gray')
        window.geometry("+%d+%d" % (100, 100))

        # icons
        self.img_run = ImageTk.PhotoImage(file='run.gif')
        self.img_mic_off = ImageTk.PhotoImage(file='mic_off.gif')
        self.img_mic_on = ImageTk.PhotoImage(file='mic_on.gif')

        self.recognizer = sr.Recognizer()
        self.recognized = ''  # recognized text

        self.model = create_model()
        self.model.load_weights('best_weights.pt')

        conf = {'samples': 30,
                'pathologies': {'ha': [0, 1], 'ex': [0, 2], 'ma': [0, 5]}}
        self.images = list(load_outofset())
        self.images += list(gen_images(conf, IH, IW))
        self.iidx = 1

        self.scale = 512 // IH  # image scale factor
        self.load_image()

        self.img_panel = self.image_panel()
        self.img_panel.grid(column=0, row=0, columnspan=2, rowspan=2,
                            sticky='wn', padx=5, pady=5)

        btn_prev = ttk.Button(window, text="<<", command=self.prev_img)
        btn_prev.grid(column=0, row=2, sticky='en', padx=5, pady=5)

        btn_next = ttk.Button(window, text=">>", command=self.next_img)
        btn_next.grid(column=1, row=2, sticky='wn', padx=5, pady=5)

        self.ent_cmnd = ttk.Entry(window, width=36)
        self.ent_cmnd.bind('<Return>', lambda e: self.execute())
        self.ent_cmnd.config(font=FONT)
        self.ent_cmnd.grid(column=2, row=0, sticky='wens', padx=5,
                           pady=5)

        btn_run = tk.Button(window, image=self.img_run, borderwidth=0,
                            command=self.execute)
        btn_run.grid(column=3, row=0, sticky='en', padx=5, pady=5)

        btn_mic = tk.Button(window, image=self.img_mic_off, borderwidth=0,
                            command=self.listen_mic)
        btn_mic.on = False
        btn_mic.grid(column=4, row=0, sticky='en', padx=5, pady=5)
        self.btn_mic = btn_mic

        self.txt_out = tk.Text(window, state='disabled',
                               bg='#414141', fg='#A6A6A6',
                               width=36, height=19)
        self.txt_out.config(font=FONT)
        self.txt_out.grid(column=2, row=1, columnspan=3, rowspan=1,
                          sticky='wens', padx=5, pady=5)

        # btn_quit = ttk.Button(window, text="quit",
        # command=self.master.destroy)
        # btn_quit.grid(column=2, row=2, sticky='wn', padx=5, pady=5)

        self.mics = sr.Microphone().list_microphone_names()
        self.device_index = 0
        self.com_mic = ttk.Combobox(window, values=self.mics)
        self.com_mic.bind('<<ComboboxSelected>>', self.select_mic)
        self.com_mic.current(self.device_index)
        self.com_mic.grid(column=2, row=2, columnspan=3,
                          sticky='we', padx=5, pady=5)

        # self.window.bind("<Key>", self.key_pressed)

    def console(self, text, append=False):
        """Write text to console"""
        self.txt_out.configure(state='normal')
        if append:
            self.txt_out.insert(tk.END, text)
        else:
            self.txt_out.delete('1.0', tk.END)
            self.txt_out.insert('1.0', text)
        self.txt_out.configure(state='disabled')

    def cmdline(self, text):
        """Write text to comand line box"""
        self.ent_cmnd.delete(0, tk.END)
        self.ent_cmnd.insert(0, text)

    def contains(self, *words):
        """Returns true if any of the words is in the recognized text"""
        return any(w for w in words if w in app.ent_cmnd.get() + ' ')

    def which_topic(self):
        """Find topic/obj/pathology in text return normalized form"""
        c = self.contains
        if c('haemorrhage', 'haemorhage', 'memor', 'hemorr', 'ha '):
            return 'haemorrhage', 'ha'
        if c('exudate', 'exit', 'accident', 'expert', 'ex '):
            return 'exudate', 'ex'
        if c('microaneurysm', 'micro', 'ma '):
            return 'microaneurysm', 'ma'
        if c('optic disc', 'disc', 'optic', 'od '):
            return 'optic disc', 'od'
        if c('fovea', 'fo '):
            return 'fovea', 'fo'
        if c('fundus', 'fu '):
            return 'fundus', 'fu'
        if c('vessel', 've '):
            return 'blood vessels', 've'
        if c('image', 'pic', 'picture', 'img ', 'annotation'):
            return 'image', None
        if c('program', 'application', 'demo'):
            return 'program', None
        return 'fundus', 'fu'

    def which_action(self):
        """Find action in text and return normalized form"""
        c = self.contains
        if c('next', 'forward'):
            return 'next'
        if c('previous', 'prev' 'back'):
            return 'prev'
        if c('count', 'how many'):
            return 'count'
        if c('show', 'mark', 'highlight', 'segment', 'where'):
            return 'show'
        if c('clear', 'delete', 'remove'):
            return 'clear'
        if c('grade', 'severity', 'level'):
            return 'grade'
        if c('explain', 'what', 'describe'):
            return 'explain'
        if c('end', 'quit', 'finish'):
            return 'quit'
        return 'show'

    def which_location(self):
        """Find location in text and return normalized form"""
        c = self.contains
        if c('upper', 'up ') and ('hemi', 'half', 'field'):
            return 'upper hemifield', 'up'
        if c('lower', 'lo ') and ('hemi', 'half', 'field'):
            return 'lower hemifield', 'lo'
        return None, None

    def action_count(self, topic, patho, loc, hem):
        if loc:
            fp = 'cnt_{0}(hem_{1}(seg_{0}(x),seg_fo(x)))'.format(patho, hem)
        else:
            fp = 'cnt_{0}(seg_{0}(x))'.format(patho)
        cnt = predict_one(self.model, fp, self.imgarr)
        cnt = round(cnt.item())
        if cnt == 1:
            answer = 'There is one %s' % topic
        else:
            answer = 'There are %d %ss' % (cnt, topic)
        if loc:
            answer += ' in the ' + loc
        say(answer)
        self.console(answer)
        self.console('\n\nFP: ' + fp, True)

    def action_show(self, topic, patho, loc, hem):
        if loc:
            fp = 'hem_{1}(seg_{0}(x),seg_fo(x))'.format(patho, hem)
        else:
            fp = 'seg_{0}(x)'.format(patho)
        mask = predict_one(self.model, fp, self.imgarr)
        self.load_image(overlay=mask)
        self.show_image(load=False)
        answer = 'Showing ' + topic
        if loc:
            answer += ' in the ' + loc
        say(answer)
        self.console(answer)
        self.console('\n\nFP: ' + fp, True)

    def action_explain(self):
        segment = lambda fp: predict_one(self.model, fp, self.imgarr)
        count = lambda fp: round(segment(fp).item())
        pathos = [('fundus', 'fu'), ('optic disc', 'od'), ('fovea', 'fo'),
                  ('vessels', 've'), ('haemorrhage', 'ha'),
                  ('microaneurysm', 'ma'), ('exudate', 'ex')]
        self.console('Found:\n')
        for topic, patho in pathos:
            n = count('cnt_{0}(seg_{0}(x))'.format(patho))
            if not n: continue
            if patho in ['fu', 'od', 'fo']:
                say('The %s is here' % topic).join()
            elif patho == 've':
                say('these are the vessels').join()
            else:
                self.console('%d : %s\n' % (n, topic), True)
                if n > 1:
                    say('and there are %d %ss here' % (n, topic)).join()
                else:
                    say('and there is one %s here' % topic).join()
            self.load_image(overlay=segment('seg_%s(x)' % patho))
            self.show_image(load=False)
            self.update()
            time.sleep(2)
        say('that is all')
        self.show_image()

    def action_grade(self):
        grade = lambda fp: predict_one(self.model, fp, self.imgarr).item()
        grades = []
        fp = 'Not(cnt_ma(seg_ma(x))+cnt_ha(seg_ha(x))+cnt_ex(seg_ex(x)))'
        grades.append(('healthy', fp, grade(fp)))
        fp = 'Gt_0(cnt_ma(seg_ma(x)))'
        grades.append(('mild', fp, grade(fp)))
        fp2 = lambda h: 'Gt_2(cnt_ma(hem_%s(seg_ma(x), seg_fo(x))))' % h
        fp = 'Xor(%s,%s)' % (fp2('up'), fp2('lo'))
        grades.append(('moderate', fp, grade(fp)))
        fp3 = lambda p: 'cnt_{0}(seg_{0}(x))'.format(p)
        fp = 'Or(%s,%s)' % (fp3('ex'), fp3('ha'))
        grades.append(('severe', fp, grade(fp)))

        for g, fp, y in reversed(grades):
            if y > 0.5:
                self.console('Grade is %s (%.1f)\n' % (g, y))
                for topic, _, cnt in self.count_pathologies():
                    self.console('\n - %s: %d' % (topic, cnt), True)
                self.console('\n\nFP: %s' % fp, True)
                if g == 'healthy':
                    say('This patient is healthy')
                else:
                    say('This is a case of %s diabetic retinopathy' % g)
                break

    def count_pathologies(self):
        pathos = [('haemorrhage', 'ha'), ('microaneurysm', 'ma'),
                  ('exudate', 'ex')]
        for topic, patho in pathos:
            fp = 'cnt_{0}(seg_{0}(x))'.format(patho)
            y = predict_one(self.model, fp, self.imgarr)
            yield topic, patho, int(y.item())

    def check_is_fundus(self):
        """Check if image has fovea and optic disc"""
        fp = 'cnt_fo(seg_fo(x)) + cnt_od(seg_od(x))'
        y = predict_one(self.model, fp, self.imgarr).item()
        return y == 2

    def execute(self):
        """Execute action given in command field"""
        topic, patho = self.which_topic()
        loc, hem = self.which_location()
        action = self.which_action()

        if action == 'quit' and topic == 'program':
            say("As you wish my master")
            self.master.destroy()
        elif action == 'next' and topic == 'image':
            say("Okay next image")
            self.next_img()
        elif action == 'clear' and topic == 'image':
            say("Clearing image")
            self.show_image()
        elif action == 'prev' and topic == 'image':
            say("Okay previous image")
            self.prev_img()
        elif not self.check_is_fundus():
            self.console("Won't fool me! That's not a fundus image!")
            say('Ha, this is not a fundus image')
        elif action == 'count':
            self.action_count(topic, patho, loc, hem)
        elif action == 'show':
            self.action_show(topic, patho, loc, hem)
        elif action == 'grade':
            self.action_grade()
        elif action == 'explain':
            self.action_explain()
        else:
            self.console('Unknown command!')
            say(choice(['What', 'Pardon me', 'Pardon', 'I do not understand']))

    def listen_mic(self):
        thread = Thread(target=speech2text, args=(self,))
        thread.start()

    def select_mic(self, event):
        name = self.com_mic.get()
        self.device_index = self.mics.index(name)
        with sr.Microphone(device_index=self.device_index) as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def next_img(self):
        self.iidx = min(self.iidx + 1, len(self.images) - 1)
        self.show_image()

    def prev_img(self):
        self.iidx = max(self.iidx - 1, 0)
        self.show_image()

    def show_image(self, load=True):
        if load:
            self.load_image()
        self.img_panel.config(image=self.image)

    def load_image(self, overlay=None):
        self.imgarr = self.images[self.iidx]
        image = self.imgarr
        if overlay is not None:
            image = image.copy() / 3
            mask = np.squeeze(overlay).astype(bool)
            image[mask] = (255, 255, 0)
        image = skt.rescale(image, scale=self.scale, order=0,
                            multichannel=True,
                            anti_aliasing=True,
                            anti_aliasing_sigma=0.5,
                            preserve_range=True).astype('uint8')
        self.image = ImageTk.PhotoImage(image=Image.fromarray(image))

    def image_panel(self):
        panel = ttk.Label(self.master, image=self.image)
        panel.bind('<ButtonPress-1>', self.img_click)
        panel.place(x=0, y=0)
        return panel

    def img_click(self, event):
        c = event.x // self.scale
        r = event.y // self.scale
        print(r, c)
        # self.images[self.iidx][r, c, :] = (0, 255, 0)
        # self.show_image()

    def key_pressed(self, event):
        print("key pressed:", event)
        if event.keycode == 39:  # cursor left
            self.next_img()
        if event.keycode == 37:  # cursor right
            self.prev_img()


if __name__ == '__main__':
    window = ThemedTk(theme='equilux')  # arc,plastik,equilux,aqua,scidgrey
    # window.wm_attributes('-fullscreen', True)
    # window.overrideredirect(1)  # no title bar
    app = App(window)
    app.mainloop()
