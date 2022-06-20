# import the necessary packages
from skimage.metrics import structural_similarity
import imutils
import cv2
import tkinter as tk
import numpy as np

class Image_Difference():
    def __init__(self) -> None:
        print("")

    def image_differences(self, imageA, imageB, pred_score, threshold):
        print("score: ", pred_score)
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()
        if pred_score < threshold:  # if frame has score lower than threshold
            # convert RGB images to grayscale images
            grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (score, diff) = structural_similarity(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            diff = cv2.GaussianBlur(diff, (7, 7), sigmaX=3, sigmaY=3)
            #cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
            #print("SSIM: {}".format(score))

            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            #thresh = cv2.threshold(diff, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w < 5 or h < 20:
                    continue

                cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
            return imgA_clone, imgB_clone, thresh
        else:
            blank_image = np.zeros((256,256,3), np.uint8)
            return imgA_clone, imgB_clone, blank_image

    def image_differences_subtract(self, imageA, imageB, pred_score, threshold):
        print("score: ", pred_score)
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()
        if pred_score < threshold:  # if frame has score lower than threshold
            # convert RGB images to grayscale images
            grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            (score, diff) = structural_similarity(grayA, grayB, full=True)
            diff = (diff * 255).astype("uint8")
            diff = cv2.GaussianBlur(diff, (7, 7), sigmaX=3, sigmaY=3)
            #cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
            #print("SSIM: {}".format(score))

            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            #thresh = cv2.threshold(diff, 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w < 5 or h < 20:
                    continue

                cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
             
            # show the output images
            #cv2.imshow("Test frame compared", cv2.resize(imageA, None, fx=1, fy=1))
            #cv2.imwrite("Result_Original.png", imageA)
            #cv2.imshow("Pred frame compared", cv2.resize(imageB, None, fx=1, fy=1))
            #cv2.imwrite("Result_Modified.png", imageB)
            #cv2.imshow("Diff", cv2.resize(diff, None, fx=1, fy=1))
            #cv2.imwrite("Result_Diff.png", diff)
            #cv2.imshow("Thresh", cv2.resize(thresh, None, fx=1, fy=1))
            #cv2.imwrite("Result_Thresh.png", thresh)
            #cv2.waitKey(0)
            return imgA_clone, imgB_clone
        else:
            #cv2.imshow("Test frame compared", cv2.resize(imageA, None, fx=1, fy=1))
            #cv2.imwrite("Result_Original.png", imageA)
            #cv2.imshow("Pred frame compared", cv2.resize(imageB, None, fx=1, fy=1))
            return imgA_clone, imgB_clone
