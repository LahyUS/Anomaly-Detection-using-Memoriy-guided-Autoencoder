from cProfile import label
import tkinter as tk
from tkinter import W, filedialog, Text
import os
import cv2 as cv
from Image_Similarity import CompareHistogram as ch
from Image_Similarity import CompareFeatures as cf
from Image_Similarity import Image_Diff as id
import numpy as np
from PIL import ImageTk, Image
#test_folder = os.getcwd()
#
#videos = OrderedDict()
#videos_list = sorted(glob.glob(os.path.join(test_folder, '*')))
#for video in videos_list:
#    video_name = video.split('/')[-1]  # split out the last string of video name
#    videos[video_name] = {}
#    videos[video_name]['path'] = video
#    videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
#    videos[video_name]['frame'].sort()
#    videos[video_name]['length'] = len(videos[video_name]['frame'])


#query = torch.randint(9, [3, 3, 1])
#keys = torch.randint(9, [3, 1])
#
#softmax_score_query, softmax_score_memory = get_score(keys, query)
#
#_, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)  # Index of the largest value in column
#_, updating_indices = torch.topk(softmax_score_query, 1, dim=0)  # Index of the largest value in row
#
#query_update = torch.zeros((3, 3))  # create a zeros tensor
#for i in range(3):
#    idx = torch.nonzero(gathering_indices.squeeze(1) == i)
#    a, _ = idx.size()
#    if a != 0:
#        query_update[i] = torch.sum(((softmax_score_query[idx, i] / torch.max(softmax_score_query[:, i]))
#                                     * query[idx].squeeze(1)), dim=0)
#    else:
#        query_update[i] = 0
#print()

# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/	

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk


LARGE_FONT= ("Verdana", 12)


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self)
        tk.Tk.wm_title(self, "Sea of BTC client")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="Graph Page",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        

app = SeaofBTCapp()
app.mainloop()