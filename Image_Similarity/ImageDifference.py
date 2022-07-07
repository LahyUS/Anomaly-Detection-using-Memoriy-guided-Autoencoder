# import the necessary packages
from skimage.metrics import structural_similarity
import imutils
import cv2
import tkinter as tk
import numpy as np

class ImageDifference():
    def __init__(self) -> None:
        print("")

    def image_differences(self, test_image, predicted_image, pred_score, threshold):
        print("score: ", pred_score)
        test_clone = test_image.copy()
        pred_clone = predicted_image.copy()

        # convert RGB images to grayscale images
        grayA = cv2.cvtColor(test_clone, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(pred_clone, cv2.COLOR_BGR2GRAY)        
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (SSIM_score, diff) = structural_similarity(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        diff = cv2.GaussianBlur(diff, (5, 5), sigmaX=3, sigmaY=3)
        #cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
        print("SSIM: {}".format(SSIM_score))

        if pred_score < threshold:  # check if frame has score lower than threshold
            # Threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            # both of these settings (BINARY_INV and OTSU) are applied at the same time using the vertical bar
        
            # Firstly, we need to separate foreground and background
            foreground_extracted = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)[1]
            # Then, inverse change pixel intensity for visualizing
            inverse_foreground = cv2.threshold(foreground_extracted, 0, 255, cv2.THRESH_BINARY_INV)
            # Finally, get the thresholded image
            thesholded_img = inverse_foreground[1]


            # Post-Process data
            #kernel_2 = np.ones((2, 2), np.uint8)
            #kernel_3 = np.ones((3, 3), np.uint8)
            #kernel_4 = np.ones((4, 4), np.uint8)
            kernel_5 = np.ones((5, 5), np.uint8)
            kernel_7 = np.ones((7, 7), np.uint8)
            kernel_11 = np.ones((11, 11), np.uint8)
            #Closing_img = cv2.morphologyEx(thesholded_img, cv2.MORPH_CLOSE, kernel=kernel_4, iterations=1)  # dilate -> erode 
            Opening_img = cv2.morphologyEx(thesholded_img, cv2.MORPH_OPEN, kernel=kernel_5, iterations=1)  # erode -> dilate
            #Eroded_img = cv2.erode(Opening_img, kernel=kernel_7, iterations=1)

            #complete_process_img = cv2.erode(Closing_img, kernel=kernel_2, iterations=1)
            complete_process_img = Opening_img
            # Next step, we find the contours of thresholded image
            cnts = cv2.findContours(complete_process_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours in order to show bbox as well as eliminate small regions 
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w < 17 or h < 17:
                    continue

                cv2.rectangle(test_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(pred_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)

            return test_clone, pred_clone, complete_process_img, thesholded_img, SSIM_score
        else:
            theshold_blank_image = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255
            return test_clone, pred_clone, theshold_blank_image, diff_blank_img, SSIM_score

    def image_differences_subtract(self, imageA, imageB):
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()
        subtracted = cv2.bitwise_xor(imgB_clone, imgA_clone)

        return subtracted

    def image_differences_pixel_label(self, imageA, imageB, pred_score, threshold, pixel_label):
        print("score: ", pred_score)
        imgA_clone = imageA.copy()
        imgB_clone = imageB.copy()
        if pred_score < threshold:  # check if frame has score lower than threshold
            # convert RGB images to grayscale images
            grayA = cv2.cvtColor(imgA_clone, cv2.COLOR_BGR2GRAY)
            grayB = cv2.cvtColor(imgB_clone, cv2.COLOR_BGR2GRAY)

            # compute the Structural Similarity Index (SSIM) between the two
            # images, ensuring that the difference image is returned
            #(score, diff) = structural_similarity(grayA, grayB, full=True)
            #diff = (diff * 255).astype("uint8")
            #diff = cv2.GaussianBlur(diff, (5, 5), sigmaX=3, sigmaY=3)
            
            #cv2.imshow("diff image", cv2.resize(diff, None, fx=1, fy=1))
            #print("SSIM: {}".format(score))

            # threshold the difference image, followed by finding contours to
            # obtain the regions of the two input images that differ
            # both of these settings (BINARY_INV and OTSU) are applied at the same time using the vertical bar
            # thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = np.uint8(pixel_label)
            # find the contours of thresholded image
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # loop over the contours
            for c in cnts:
                # compute the bounding box of the contour and then draw the
                # bounding box on both input images to represent where the two
                # images differ
                (x, y, w, h) = cv2.boundingRect(c)

                # Remove regions in-significant regions
                if w < 20 or h < 20:
                    continue

                cv2.rectangle(imgA_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.rectangle(imgB_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)

            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255
            return imgA_clone, imgB_clone, thresh, diff_blank_img

        else:
            theshol_blank_image = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img = np.zeros((256, 256, 3), np.uint8)
            diff_blank_img += 255
            return imgA_clone, imgB_clone, theshol_blank_image, diff_blank_img

    def compare_image_diff(self, pixel_label, SSIM_diff_image):
        pixel_thresh = np.uint8(pixel_label)
        cv2.imshow("pixel_labels", cv2.resize(pixel_thresh, (256, 256)))
        cv2.imshow("SSIM_diff", cv2.resize(SSIM_diff_image, (256, 256)))
        
        # Post-Process data
        kernel_2 = np.ones((2, 2), np.uint8)
        kernel_3 = np.ones((3, 3), np.uint8)
        kernel_4 = np.ones((4, 4), np.uint8)
        kernel_5 = np.ones((5, 5), np.uint8)
        kernel_6 = np.ones((6, 6), np.uint8)
        kernel_7 = np.ones((7, 7), np.uint8)

        Opening_img = cv2.morphologyEx(SSIM_diff_image, cv2.MORPH_OPEN, kernel=kernel_7, iterations=1)  # erode -> dilate
        Closing_img = cv2.morphologyEx(Opening_img, cv2.MORPH_CLOSE, kernel=kernel_4, iterations=1)  # dilate -> erode 
        #Eroded_img = cv2.erode(Closing_img, kernel=kernel_4, iterations=1)
        complete_process_img = Closing_img
        cv2.imshow("Eroded_img", cv2.resize(complete_process_img, (256, 256)))

        grayscale_cvt_pixel = cv2.threshold(pixel_label, 1, 255, cv2.THRESH_BINARY_INV)[1]
        grayscale_cvt_SSIM = cv2.threshold(complete_process_img, 1, 255, cv2.THRESH_BINARY_INV)[1]
        #cv2.imshow("cvt_pixel", cv2.resize(grayscale_cvt_pixel, (256, 256)))
        #cv2.imshow("cvt_SSIM", cv2.resize(grayscale_cvt_SSIM, (256, 256)))

        subtracted_img = self.image_differences_subtract(complete_process_img, pixel_thresh)
        (score, diff_img) = structural_similarity(grayscale_cvt_pixel, grayscale_cvt_SSIM, full=True)
        print("SSIM score: ", score)
        #diff_img = (diff_img * 255).astype("uint8")
        cv2.imshow("compared_diff", cv2.resize(subtracted_img, (256, 256)))
        cv2.waitKey(0)
