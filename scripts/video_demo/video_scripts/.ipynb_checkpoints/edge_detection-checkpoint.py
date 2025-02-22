import cv2
import numpy as np
from os.path import join, splitext
from os import listdir
import pandas as pd
import os


def edge_detection(data_structure, n: int):
    """
    edge_detection reads the data as defined in data_structure and measures the center of a circle vs time.
    The results are outputted to a .csv file with the path defined in data_structure

    """

    # These parameters will be part of the function form input
    FNAME = data_structure.data_folders[n]
    DATA_DIR = data_structure.data_path + "/" + FNAME
    SAVE_DIR = data_structure.save_path
    ext = data_structure.extension

    # Generate a list of all image files in the target folder
    im_list = sorted(splitext(f)[0] for f in listdir(DATA_DIR) if f.endswith(ext))

    # x,y coordinates and radius
    x = np.zeros(len(im_list))
    y = np.zeros(len(im_list))
    r = np.zeros(len(im_list))

    for i in np.arange(0, len(im_list), 1):
        # read in image and convert to grayscale
        img = cv2.imread(join(DATA_DIR, im_list[i] + ext), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # use Hough transform to find circle
        detected_circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=1,
            maxRadius=40,
        )

        # write data from circle to array
        if detected_circles is not None:
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, c = pt[0], pt[1], pt[2]
                x[i] = a
                y[i] = b
                r[i] = c

    # Convert arrays to datatable
    d = {"x": x, "y": y, "r": r}
    data_table = pd.DataFrame(data=d)

    # Output to .csv
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    data_table.to_csv((SAVE_DIR + "/" + FNAME + ".csv"), index=False)
