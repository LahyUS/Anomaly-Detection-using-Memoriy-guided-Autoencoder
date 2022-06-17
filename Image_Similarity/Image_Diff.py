# import the necessary packages
from skimage.metrics import structural_similarity
import imutils
import cv2


def image_differences(imageA, imageB):
    # convert RGB images to grayscale images
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # show the output images
    cv2.imshow("Original", cv2.resize(imageA, None, fx=1, fy=1))
    #cv2.imwrite("Result_Original.png", imageA)
    cv2.imshow("Modified", cv2.resize(imageB, None, fx=1, fy=1))
    #cv2.imwrite("Result_Modified.png", imageB)
    cv2.imshow("Diff", cv2.resize(diff, None, fx=1, fy=1))
    #cv2.imwrite("Result_Diff.png", diff)
    cv2.imshow("Thresh", cv2.resize(thresh, None, fx=1, fy=1))
    #cv2.imwrite("Result_Thresh.png", thresh)
    cv2.waitKey(0)
