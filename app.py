from cProfile import label
import tkinter as tk
from tkinter import ttk
from tkinter import W, filedialog, Text
import os
import cv2
from matplotlib.image import FigureImage
from sklearn import utils
from Image_Similarity import Image_Diff as id
import numpy as np
from PIL import ImageTk, Image
import time
from sklearn import metrics
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import ( FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.animation import FuncAnimation
import pandas as pd 

LARGE_FONT= ("Verdana", 12)

def mini_frame_coord(window_H, window_W, frame_h, frame_w):
    minus_h = window_H - frame_h
    minus_w = window_W - frame_w
    bias_h = minus_h/2
    bias_w = minus_w/2
    return bias_h, bias_w

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window  # window = tk.Tk()
        #self.window.title(window_title)
        self.window.iconbitmap()
        self.window.wm_title("Anomaly Detection Application")

        # open video source (by default this will try to open the computer webcam)
        self.video_source = video_source
        self.vid = VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas_H, self.canvas_W = 300, 900
        self.canvas_center_H, self.canvas_center_W = self.canvas_H/2, self.canvas_W/2
        if self.video_source == 0:
            self.canvas = tk.Canvas(window, width = self.vid.width, height = self.vid.height)
        else:
            self.canvas = tk.Canvas(window, width = self.canvas_W, height = self.canvas_H, background="#4E747E")
        self.canvas.pack()
        
        # Button that lets the user take a snapshot
        self.btn_snapshot=tk.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Create an ImgDiff object for do difference on images
        self.ImgDiff = id.Image_Difference()

        # bias height and bias width for add mini frames canvas
        self.bias_h, self.bias_w = mini_frame_coord(self.canvas_H, self.canvas_W, 256, 256)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.iter_frame = 0
        # used to record the time when we processed last frame
        self.prev_frame_time = 0
        # used to record the time at which we processed current frame
        self.new_frame_time = 0

        if video_source == 0:   
            self.update()
        else:
            #self.show_figure_of_scores_on_frame()
            self.static_update()
            self.static_update_figure()

        #self.window.after(self.delay, self.static_update)
        #self.window.after(self.delay, self.static_update_figure)
        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        ## Get a frame from the video source
        #ret, frame = self.vid.get_frame()
#
        #if ret:
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

        # Calculate difference image
        test_img, pred_img, diff_img = self.ImgDiff.image_differences(test_frame, predicted_frame, anomaly_score, self.vid.opt_threshold)

        # Closes all the frames time when we finish processing for this frame
        self.new_frame_time = time.time()

        # FPS will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1/(self.new_frame_time - self.prev_frame_time)
        self.prev_frame_time = self.new_frame_time
        fps = "fps: " + str(int(fps))
        cv2.putText(test_img, fps, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # convert opencv narray image to PIL image
        self.photo_test = ImageTk.PhotoImage(image = Image.fromarray(test_img))
        self.photo_pred = ImageTk.PhotoImage(image = Image.fromarray(pred_img))
        self.photo_diff = ImageTk.PhotoImage(image = Image.fromarray(diff_img))

        # attach test, predicted, difference images on canvas
        self.canvas.create_image(35, self.bias_h, image = self.photo_test, anchor = tk.NW)
        self.canvas.create_image(35+256+35, self.bias_h, image = self.photo_pred, anchor = tk.NW)
        self.canvas.create_image(35+256+35+256+35, self.bias_h, image = self.photo_diff, anchor = tk.NW)

        self.window.after(self.delay, self.static_update)
        if self.iter_frame == 170:
            self.iter_frame = 0
        else:
            self.iter_frame+=1

    def show_figure_of_scores_on_frame(self):
        # create a figure member
        figure = self.get_anomaly_scores_figure(self.vid.frame_scores, self.vid.labels,
                                     "Ped2", "Pred", "Trained on Ped2")

        #frame = tk.Frame(self.container)
        label = tk.Label(text="Graph Page!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
    
        canvas = FigureCanvasTkAgg(figure)
        #canvas.draw()

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
        anomaly_score_total_list = np.load("./datasets/predicted/anomaly_score.npy")
        y_score = np.squeeze(anomaly_score_total_list[:self.iter_frame+1])

        len = y_score.size
        x = np.arange(0, len)

        # Declare a clear axis each time  
        plt.cla()
        y_thresh = self.vid.opt_threshold
        # create a legend
        x_thresh = (0, 175)
        y_thresh = (y_thresh, y_thresh)
        plt.plot(x, y_score, color="steelblue", label='score/frame')
        plt.plot(x_thresh, y_thresh, color="red", marker = 'o', label="threshold")
        plt.legend(loc='upper right')
        plt.tight_layout()
        #return plt.gcf()

    def static_update_figure(self):
        self.figure = plt.figure()            
        self.ax = self.figure.add_subplot(111)
        # Set label for the figure
        label = tk.Label(text="Anomaly Score Graph!", font=LARGE_FONT)
        label.pack(pady=10, padx=10)
        self.ani = FuncAnimation(self.figure, self.static_animate, interval=1000)
        # Create canvas that hold figure
        self.canvas_fig = FigureCanvasTkAgg(plt.gcf(), self.window)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #self.window.after(self.delay, self.static_update_figure)

class VideoCapture:
    def __init__(self, video_source=0):
        self.video_source = video_source
        # Open the video source, # capture video by webcam by default 
        if video_source == 0:
            self.vid = cv2.VideoCapture(video_source)  
            if not self.vid.isOpened():
                raise ValueError("Unable to open video source", video_source)
                    
            # Get video source width and height
            self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        else:
            # load test and predicted frames
            test_frames, predicted_framese = self.get_dataset_frames()
            self.vid = [test_frames, predicted_framese]
            # load frame
            self.frame_scores = np.load("./datasets/predicted/anomaly_score.npy")
            self.labels = np.load('./data/frame_labels_'+'ped2'+'.npy')
            self.opt_threshold = self.optimalThreshold(self.frame_scores, self.labels)

    def get_frame(self):
        #if self.vid.isOpened():
        #    ret, frame = self.vid.read()
        #    if ret:
        #        # Return a boolean success flag and the current frame converted to BGR
        #        return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #    else:
        #        return (ret, None)
        #else:
        #    return (ret, None)
        pass
#
    def get_static_frame(self, iter_frame):
        # load the two input images
        i = iter_frame
        list_imageA = self.vid[0]  # test frame
        imageA = list_imageA[i+4]
        list_imageB = self.vid[1]
        imageB = list_imageB[i]    # pred frame

        # resize image
        w1, h1, c1 = imageA.shape
        w2, h2, c2 = imageB.shape
        if w1 != 256:
            imageA = cv2.resize(imageA, (256, 256))
        if w2 != 256:
            imageB = cv2.resize(imageB, (256, 256))

        anomaly_score = self.frame_scores[i]
        return imageA, imageB, anomaly_score

    def get_dataset_frames(self):
        time_t = 0
        test_input_path = []
        for i in range(178):
            frame_th = ''
            frame_num = time_t + i
            if frame_num < 10:
                frame_th = '00' + str(frame_num)
            elif frame_num < 100:
                frame_th = '0' + str(frame_num)
            else:
                frame_th = str(frame_num)
            test_input_path.append("./datasets/testing/frames/01/" + frame_th + ".jpg")

        pred_input_path = []
        for i in range(174):
            frame_th = ''
            frame_num = time_t + i
            if frame_num < 10:
                frame_th = '00' + str(frame_num)
            elif frame_num < 100:
                frame_th = '0' + str(frame_num)
            else:
                frame_th = str(frame_num)
            pred_input_path.append("./datasets/predicted/pred/frames/0" + frame_th + ".jpg")

        test_input_imgs = []
        for i in range(178):
            img = cv2.imread(test_input_path[i])
            test_input_imgs.append(img)

        pred_input_imgs = []
        for i in range(174):
            img = cv2.imread(pred_input_path[i])
            pred_input_imgs.append(img)

        return test_input_imgs, pred_input_imgs

    def optimalThreshold(self, anomal_scores, labels):
            y_true = 1 - labels[0, :1962]
            y_true  = np.squeeze(y_true)
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
        if self.video_source == 0:
            if self.vid.isOpened():
                self.vid.release()
        else:
            pass

# Create a window and pass it to the Application object
App(tk.Tk(), "Tkinter and OpenCV", video_source=1)
cv2.destroyAllWindows()