# import the necessary packages
import matplotlib.pyplot as plt
import cv2 as cv


def compareHistogram(src_test1, src_test2):
    # Convert to RGB image
    src_base = src_test1.copy()
    rgb_base_img = cv.cvtColor(src_base, cv.COLOR_BGR2RGB)
    rgb_img_1 = cv.cvtColor(src_test1, cv.COLOR_BGR2RGB)
    rgb_img_2 = cv.cvtColor(src_test2, cv.COLOR_BGR2RGB)
    images = []
    images.append(rgb_img_1)
    images.append(rgb_img_2)

    # extract a 3D RGB color histogram from the image,
    # using 256 bins per channel, normalize, and update the index
    bins = 256
    histSize = [bins, bins, bins]  # [Red_bins, Green_bins, Blue_bins]
    ranges = [0, 256, 0, 256, 0, 256]
    channels = [0, 1, 2]

    hist_list = []
    # Calculate histogram for each image
    base_hist = cv.calcHist([rgb_base_img], channels, None, histSize, ranges, accumulate=False)
    hist_1 = cv.calcHist([rgb_img_1], channels, None, histSize, ranges, accumulate=False)
    hist_2 = cv.calcHist([rgb_img_2], channels, None, histSize, ranges, accumulate=False)

    # Normalize histogram
    cv.normalize(base_hist, base_hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    cv.normalize(hist_1, hist_1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    cv.normalize(hist_2, hist_2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    # Add histograms to hist_list
    hist_list.append(hist_1)
    hist_list.append(hist_2)

    # initialize OpenCV methods for histogram comparison
    OPENCV_METHODS = (
        ("Correlation", cv.HISTCMP_CORREL),
        ("Chi-Squared", cv.HISTCMP_CHISQR),
        ("Intersection", cv.HISTCMP_INTERSECT),
        ("Hellinger", cv.HISTCMP_BHATTACHARYYA))

    # loop over the comparison methods
    for (methodName, method) in OPENCV_METHODS:
        # initialize the results dictionary and the sort
        # direction
        results = []

        for i in range(len(hist_list)):
            d = cv.compareHist(base_hist, hist_list[i], method)
            results.append(d)

        # show the query image
        fig = plt.figure("Query")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(rgb_base_img)
        plt.axis("off")

        # initialize the results figure
        fig = plt.figure("Results: %s" % (methodName))
        fig.suptitle(methodName, fontsize=20)
        # loop over the results
        for i in range(len(results)):
            # show the result
            ax = fig.add_subplot(1, 2, i + 1)
            ax.set_title("%.2f" % results[i])
            plt.imshow(images[i])
            plt.axis("off")

    # show the OpenCV methods
    plt.show()
