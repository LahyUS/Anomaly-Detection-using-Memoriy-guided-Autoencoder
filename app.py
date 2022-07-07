import argparse
from this import s
import threading
import pandas as pd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter import filedialog, Text
from matplotlib.animation import FuncAnimation
from cProfile import label
import tkinter as tk
from tkinter import ttk
from tkinter import W, filedialog, Text
import os
import cv2
from matplotlib.image import FigureImage
from sklearn import utils
from torch import le, sub
from image_similarity import ImageDifference as id
import numpy as np
from PIL import ImageTk, Image
import time
from sklearn import metrics
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

LARGE_FONT = ("Verdana", 12)

DEFAULT_DATASET_NAME = "ped2"
DEFAULT_METHOD = "pred"
DEFAULT_FRAME_PRED_INPUT = 4


def mini_frame_coord(window_H, window_W, frame_h, frame_w):
    minus_h = window_H - frame_h
    minus_w = window_W - frame_w
    bias_h = minus_h/2
    bias_w = minus_w/2
    return bias_h, bias_w


class App:
    def __init__(self, window, window_title, dataset_type=DEFAULT_DATASET_NAME, method=DEFAULT_METHOD, video_source=0):
        # create a window that contains everything on it
        self.window = window  # window = tk.Tk()
        self.window.iconbitmap()
        # set title for window
        self.window.wm_title("Anomaly Detection Application")
        self.dataset_type = dataset_type
        self.method = method

        # create canvas to show widget
        self.set_up_canvas(video_source)

        # create button for file loading process
        self.setup_Buttons()

        # create a video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(video_source, self.current_data_path, self.dataset_type)
        self.numPredFrame = len(self.vid.vid[1])

        # create an ImgDiff object for do difference on images
        self.ImgDiff = id.ImageDifference()

        # update ktinker canvas
        if video_source == 0:
            self.update()
        else:
            self.static_update()
            threading.Thread(target=self.static_update_figure()).start()

        self.window.mainloop()

    def set_up_canvas(self, type):
        # Create a canvas that can fit the above video source size
        self.canvas_H, self.canvas_W = 350, 1200
        self.canvas_center_H, self.canvas_center_W = self.canvas_H/2, self.canvas_W/2
        if type == 0:
            self.canvas = tk.Canvas(
                self.window, width=self.vid.width, height=self.vid.height)
        else:
            self.canvas = tk.Canvas(
                self.window, width=self.canvas_W, height=self.canvas_H, background="#4E747E")
        self.canvas.pack()

        # Setup for timer
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.iter_frame = 0
        self.prev_frame_time = 0  # used to record the time when we processed last frame
        self.new_frame_time = 0  # used to record the time at which we processed current frame

        # Set up for showing frames
        # bias height and bias width for add mini frames canvas
        self.bias_h, self.bias_w = mini_frame_coord(
            self.canvas_H, self.canvas_W, 256, 256)
        self.frame_1_x_axis = 35
        self.frame_2_x_axis = 35+256+35
        self.frame_3_x_axis = 35+256+35+256+35
        self.frame_4_x_axis = 35+256+35+256+35+256+35
        self.fps = self.canvas.create_text(
            450, 320, fill="white", font="Times 15 italic bold", text="fps: ")
        self.frame_th = self.canvas.create_text(
            550, 320, fill="white", font="Times 15 italic bold", text="frame: ")
        self.anomaly_tag = self.canvas.create_text(
            700, 320, fill="white", font="Times 20 italic bold", text="Abnormal: ")
        self.dataset_name_tag = self.canvas.create_text(
            35, 320, fill="white", font='Helvatica 20 bold', text="Dataset: Ped2", anchor=tk.NW)

        # Show name of each kind of showing frame
        self.canvas.create_text(self.frame_1_x_axis, 10, fill="white",
                                font='Helvatica 20 bold', text="Ground-truth", anchor=tk.NW)
        self.canvas.create_text(self.frame_2_x_axis, 10, fill="white",
                                font='Helvatica 20 bold', text="Predicted", anchor=tk.NW)
        self.canvas.create_text(self.frame_3_x_axis, 10, fill="white",
                                font='Helvatica 20 bold', text="Difference", anchor=tk.NW)
        self.canvas.create_text(self.frame_4_x_axis, 10, fill="white",
                                font='Helvatica 20 bold', text="Anomaly Regions", anchor=tk.NW)

    def change_dataset(self):
        # filename = filedialog.askopenfilename(
        #    initialdir="/", title="Select file", filetypes=(("image", "*.jpg"), ("image", "*.png"), ("video", "*.avi"), ("all files", "*.*")))
        folder_path = filedialog.askdirectory()
        self.current_data_path.append(folder_path)
        current_dataset_name = folder_path.split('/')[-1]

        self.iter_frame = 0
        self.vid = VideoCapture(1, self.current_data_path, current_dataset_name)
        self.numPredFrame = len(self.vid.vid[1])
        self.canvas.itemconfig(self.dataset_name_tag, fill='white',
                               text="Dataset: {}".format(current_dataset_name))

    def setup_Buttons(self):
        self.current_data_path = ['./dataset/' + self.dataset_type + '/', ]
        # Button that lets the user take a snapshot
        self.open_dataset = tk.Button(self.window, text="Open Dataset", width=50, command=self.change_dataset,
                                      fg='white', bg="#263D42")
        self.open_dataset.pack(anchor=tk.CENTER, expand=True)

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") +
                        ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        #ret, frame = self.vid.get_frame()
        #
        # if ret:
        #    # convert opencv narray image to PIL image
        #    self.photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
        #    # attach image on canvas
        #    self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        #
        #self.window.after(self.delay, self.update)
        pass

    def static_update(self):
        # Get a frame from the video source
        test_frame, predicted_frame, anomaly_score = self.vid.get_static_frame(self.iter_frame)
        optimal_threshold = self.vid.opt_threshold
        # Calculate difference image
        test_img_detected, pred_img_detected, thresholded_img, SSIM_diff_img, SSIM_score = self.ImgDiff.image_differences(
            test_frame, predicted_frame, anomaly_score, self.vid.opt_threshold)

        # Closes all the frames time when we finish processing for this frame
        self.new_frame_time = time.time()
        # Calculate frame per seccond value
        fps = 1/(self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time

        # Show information on canvas
        self.canvas.itemconfig(self.fps, text="fps: {}".format(int(fps)))  # Update fps on canvas
        self.canvas.itemconfig(self.frame_th, text="frame: {}".format(self.iter_frame))
        if anomaly_score < optimal_threshold:
            self.canvas.itemconfig(
                self.anomaly_tag, fill='red', text="Abnormal: YES")
        else:
            self.canvas.itemconfig(
                self.anomaly_tag, fill='white', text="Abnormal: NO")

        # Convert opencv narray images to PIL images
        self.photo_test = ImageTk.PhotoImage(image=Image.fromarray(test_frame))
        self.photo_pred = ImageTk.PhotoImage(image=Image.fromarray(predicted_frame))
        self.photo_diff = ImageTk.PhotoImage(image=Image.fromarray(thresholded_img))
        self.detected_regions = ImageTk.PhotoImage(image=Image.fromarray(test_img_detected))

        # Attach test, predicted, difference and detected_regions images on canvas
        self.canvas.create_image(self.frame_1_x_axis, self.bias_h, image=self.photo_test, anchor=tk.NW)
        self.canvas.create_image(self.frame_2_x_axis, self.bias_h, image=self.photo_pred, anchor=tk.NW)
        self.canvas.create_image(self.frame_3_x_axis, self.bias_h, image=self.photo_diff, anchor=tk.NW)
        self.canvas.create_image(self.frame_4_x_axis, self.bias_h, image=self.detected_regions, anchor=tk.NW)

        # Function callback
        self.window.after(50, self.static_update)
        if self.iter_frame == len(self.vid.vid[1]):
            self.iter_frame = 0
        else:
            self.iter_frame += 1

    def show_figure_of_scores_on_frame(self):
        # create a figure member
        figure = self.get_anomaly_scores_figure(self.vid.frame_scores, self.vid.labels,
                                                self.dataset_type, self.method, "Trained on Ped2")

        #frame = tk.Frame(self.container)
        label = tk.Label(text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        canvas = FigureCanvasTkAgg(figure)
        # canvas.draw()

        # attach what is created
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def get_anomaly_scores_figure(self, anomaly_score_total_list, labels, dataset_type, method, trained_model_using):
        matrix = np.array([labels == 1])

        # Mask the False occurences in the numpy array as 'bad' data
        matrix = np.ma.masked_where(matrix == True, matrix)

        # Create a ListedColormap with only the color green specified
        cmap = colors.ListedColormap(['none'])

        # Use the `set_bad` property of `colormaps` to set all the 'bad' data to red
        cmap.set_bad(color='lavenderblush')
        fig, ax = plt.subplots()
        fig.set_size_inches(7, 4)
        plt.title('Anomaly score/frame, method: ' + method + ', dataset: ' + dataset_type +
                  ', trained model used: ' + trained_model_using)
        #ax.pcolormesh(matrix, cmap=cmap, edgecolor='none', linestyle='-', lw=1)

        y = anomaly_score_total_list
        x = np.arange(0, len(y))
        plt.plot(x, y, color="steelblue", label="score/frame")
        plt.legend(loc='lower right')  # specific location
        plt.ylabel('Score')
        plt.xlabel('Frames')
        return fig

    def animate(self, i):
        data = pd.read_csv('data.csv')
        x = data['x_value']
        y1 = data['total_1']
        y2 = data['total_2']

        # Declare a clear axis each time
        plt.cla()

        # create a legend
        plt.plot(x, y1, label='Channel 1')
        plt.plot(x, y2, label='Channel 2')
        plt.legend(loc='upper left')
        plt.tight_layout()
        return plt.gcf()

    def static_animate(self, i):
        y_score = np.squeeze(self.vid.frame_scores[:self.iter_frame+1])

        len = y_score.size
        x = np.arange(0, len)

        # Declare a clear axis each time
        plt.cla()
        y_thresh = self.vid.opt_threshold
        # create a legend

        x_thresh = (0, self.numPredFrame)
        y_thresh = (y_thresh, y_thresh)
        plt.plot(x, y_score, color="steelblue", label='score/frame')
        plt.plot(x_thresh, y_thresh, color="red",
                 marker='o', label="threshold")
        plt.plot(len, y_score[len-1], color="green",
                 marker='o', label="current_frame")
        plt.legend(loc='lower left')
        plt.ylabel('Score')
        plt.xlabel('Frames')
        plt.tight_layout()

    def static_update_figure(self):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        # Set label for the figure
        label = tk.Label(text="Anomaly Score Graph", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        self.ani = FuncAnimation(
            self.figure, self.static_animate, interval=1000)
        # Create canvas that hold figure
        self.canvas_fig = FigureCanvasTkAgg(plt.gcf(), self.window)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #self.window.after(self.delay, self.static_update_figure)


class VideoCapture:
    def __init__(self, video_type=0, data_path=[], dataset_type=DEFAULT_DATASET_NAME, frame_sequence_length=DEFAULT_FRAME_PRED_INPUT):
        self.data_path = data_path
        self.type = video_type
        self.dataset_type = dataset_type
        self.frame_sequence_length = frame_sequence_length
        # Open the video source, # capture video by webcam by default
        if self.type == 0:
            self.vid = cv2.VideoCapture(self.type)
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", self.type)

            # Get video source width and height
            self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        else:
            # load test and predicted frames
            test_frames, predicted_framese, test_num_video_index = self.get_dataset_frames()
            self.vid = [test_frames, predicted_framese, test_num_video_index]
            # load frame
            self.frame_scores = np.load(self.data_path[-1] + 'output/anomaly_score.npy')
            self.labels = np.load('./data/frame_labels_' + self.dataset_type + '.npy')
            self.opt_threshold = self.optimalThreshold(self.frame_scores, self.labels)

    def get_frame(self):
        # if self.vid.isOpened():
        #    ret, frame = self.vid.read()
        #    if ret:
        #        # Return a boolean success flag and the current frame converted to BGR
        #        return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #    else:
        #        return (ret, None)
        # else:
        #    return (ret, None)
        pass

    def get_static_frame(self, iter_frame):
        test_video_index = self.vid[2]
        # load the two input images
        i = iter_frame
        test_frame_list = self.vid[0] 

        # Get predicted frame
        pred_frame_list = self.vid[1] 
        current_pred_frame = pred_frame_list[i]

        # Get ground-truth frame
        map_index = test_video_index[i]
        true_index_of_test_frame = i + self.frame_sequence_length * test_video_index[i+map_index*4]
        current_test_frame = test_frame_list[true_index_of_test_frame]

        # resize image
        w1, h1, c1 = current_test_frame.shape
        w2, h2, c2 = current_pred_frame.shape
        if w1 != 256:
            current_test_frame = cv2.resize(current_test_frame, (256, 256))
        if w2 != 256:
            current_pred_frame = cv2.resize(current_pred_frame, (256, 256))

        anomaly_score = self.frame_scores[i]
        return current_test_frame, current_pred_frame, anomaly_score

    def get_dataset_frames(self):
        time_t = 0
        test_input_path = []
        test_video_dir = []
        test_video_dir_distinct = []
        current_path = self.data_path[-1]
        real_path = current_path + '/testing/frames'
        for path, _, files in os.walk(real_path):
            for name in files:
                if(path not in test_video_dir_distinct):
                    test_video_dir_distinct.append(path)
                test_input_path.append(os.path.join(path, name))
                test_video_dir.append(path)
        test_input_path.sort()
        test_video_dir.sort()
        test_video_dir_distinct.sort()

        test_video_dir_distinct_map_index = {}
        for i in range(len(test_video_dir_distinct)):
            test_video_dir_distinct_map_index[test_video_dir_distinct[i]] = i + 1

        for i in range(len(test_video_dir)):
            test_video_dir[i] = test_video_dir_distinct_map_index[test_video_dir[i]]

        pred_input_path = []
        for path, _, files in os.walk(self.data_path[-1] + 'output/frames'):
            for name in files:
                pred_input_path.append(os.path.join(path, name))
        pred_input_path.sort()

        test_input_imgs = []
        for i in range(len(test_input_path)):
            img = cv2.imread(test_input_path[i])
            test_input_imgs.append(img)

        pred_input_imgs = []
        for i in range(len(pred_input_path)):
            img = cv2.imread(pred_input_path[i])
            pred_input_imgs.append(img)

        return test_input_imgs, pred_input_imgs, test_video_dir

    def optimalThreshold(self, anomal_scores, labels):
        y_true = 1 - labels[0, :1962]
        y_true = np.squeeze(y_true)
        y_score = np.squeeze(anomal_scores[:1962])
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
        frame_auc = metrics.roc_auc_score(y_true, y_score)
        print("AUC: {}".format(frame_auc))
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (threshold[ix], gmeans[ix]))
        return threshold[ix]

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.type == 0:
            if self.vid.isOpened():
                self.vid.release()
        else:
            pass


# Create a window and pass it to the Application object
parser = argparse.ArgumentParser(description="anomaly detection using aemem")
parser.add_argument('--method', type=str, default='pred',
                    help='The target task for anoamly detection')
parser.add_argument('--t_length', type=int, default=5,
                    help='length of the frame sequences')
parser.add_argument('--dataset_type', type=str, default='ped2',
                    help='type of dataset: ped1, ped2, avenue, shanghai')

args = parser.parse_args()
App(tk.Tk(), "Tkinter and OpenCV", dataset_type=args.dataset_type, video_source=1)
cv2.destroyAllWindows()
