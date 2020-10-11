#BGR to HSV , HSV - hue saturation value / can easily compare
#minimum enclosing circle - minimum possible circle for the object area
#moments to find center of the area - objects center
#drawing circle - draw the minimum enclosing circle
#block diagram: reading from camera -> pre-processing image -> finding counters -> drawing minimum enclosing circle
# -> finding center of counter area -> drawing circle and center -> direction based on radius and position
import time

import imutils
import numpy as np
import cv2
from mss import mss
from PIL import Image

ok = False

def colorCalibrationMethod():
    # Trackbar onChange
    def nothing(x):
        pass

    # Create new windows
    colour = np.zeros((50, 300, 3), np.uint8)
    cv2.namedWindow('Colour Adjustment')
    cv2.namedWindow('Capture Area')
    cv2.imshow('Colour Adjustment', colour)

    # Create trackbars
    cv2.createTrackbar('X', 'Capture Area', 0, 1000, nothing)
    cv2.createTrackbar('Y', 'Capture Area', 0, 1000, nothing)
    cv2.createTrackbar('Width', 'Capture Area', 0, 500, nothing)
    cv2.createTrackbar('Height', 'Capture Area', 0, 500, nothing)
    cv2.createTrackbar('R High', 'Colour Adjustment', 255, 255, nothing)
    cv2.createTrackbar('G High', 'Colour Adjustment', 255, 255, nothing)
    cv2.createTrackbar('B High', 'Colour Adjustment', 255, 255, nothing)
    cv2.createTrackbar('R Low', 'Colour Adjustment', 0, 255, nothing)
    cv2.createTrackbar('G Low', 'Colour Adjustment', 0, 255, nothing)
    cv2.createTrackbar('B Low', 'Colour Adjustment', 0, 255, nothing)

    # Create memory of trackbar values to register future changes
    or1 = cv2.getTrackbarPos('R High', 'Colour Adjustment')
    og1 = cv2.getTrackbarPos('G High', 'Colour Adjustment')
    ob1 = cv2.getTrackbarPos('B High', 'Colour Adjustment')
    or2 = cv2.getTrackbarPos('R Low', 'Colour Adjustment')
    og2 = cv2.getTrackbarPos('G Low', 'Colour Adjustment')
    ob2 = cv2.getTrackbarPos('B Low', 'Colour Adjustment')

    # Initialize variables
    textColour = 255
    textScale = 0.7
    textFont = cv2.FONT_HERSHEY_SIMPLEX
    highhsv = [0, 0, 0]
    lowhsv = [0, 0, 0]
    sct = mss()

    while (1):
        # Get current HSV values
        r1 = cv2.getTrackbarPos('R High', 'Colour Adjustment')
        g1 = cv2.getTrackbarPos('G High', 'Colour Adjustment')
        b1 = cv2.getTrackbarPos('B High', 'Colour Adjustment')
        r2 = cv2.getTrackbarPos('R Low', 'Colour Adjustment')
        g2 = cv2.getTrackbarPos('G Low', 'Colour Adjustment')
        b2 = cv2.getTrackbarPos('B Low', 'Colour Adjustment')
        # Get current capture area
        x = cv2.getTrackbarPos('X', 'Capture Area')
        y = cv2.getTrackbarPos('Y', 'Capture Area')
        width = cv2.getTrackbarPos('Width', 'Capture Area')
        height = cv2.getTrackbarPos('Height', 'Capture Area')
        # Change color bar on changing trackbar (high or low)
        if or1 != r1 or og1 != g1 or ob1 != b1:
            or1 = r1
            og1 = g1
            ob1 = b1
            colour[:] = [b1, g1, r1]
            highhsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
            highhsv = [int(highhsv[0][0][0]), int(highhsv[0][0][1]), int(highhsv[0][0][2])]
        elif or2 != r2 or og2 != g2 or ob2 != b2:
            or2 = r2
            og2 = g2
            ob2 = b2
            colour[:] = [b2, g2, r2]
            lowhsv = cv2.cvtColor(colour, cv2.COLOR_BGR2HSV)
            lowhsv = [int(lowhsv[0][0][0]), int(lowhsv[0][0][1]), int(lowhsv[0][0][2])]
        cv2.imshow('Colour Adjustment', colour)
        # Capture screen
        img = Image.open(r"screenshot.jpg")
        img_np = np.array(img)
        # Convert and find colours from captured images
        hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        higher = np.array(highhsv)
        lower = np.array(lowhsv)
        mask = cv2.inRange(hsv, lower, higher)
        res = cv2.bitwise_and(rgb, rgb, mask=mask)
        # Display HSV over captured image
        cv2.putText(res, 'High HSV: ' + str(highhsv), (5, 210 + height), textFont, textScale,
                    (textColour, textColour, textColour), 2)
        cv2.putText(res, 'Low HSV: ' + str(lowhsv), (5, 240 + height), textFont, textScale,
                    (textColour, textColour, textColour), 2)
        # Show image with only specified color range
        cv2.imshow('Found Colours', res)
        # Adjust window sizes
        cv2.resizeWindow('Colour Adjustment', 400, 50)
        cv2.resizeWindow('Capture Area', 300, 0)
        # Break loop with esc key
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

def saveImage():
    cam = cv2.VideoCapture(0)
    time.sleep(1)
    firstFrame = None
    area = 500

    while True:
        _, img = cam.read()
        text = "Normal"
        img = imutils.resize(img, width=500)
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
        if firstFrame is None:
            firstFrame = gaussianImg
            continue
        imgDiff = cv2.absdiff(firstFrame, grayImg)
        threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
        threshImg = cv2.dilate(threshImg, None, iterations=2)
        cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            if cv2.contourArea(c) < area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Moving Object Detected"
        print(text)
        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("my_cam", img)
        key = cv2.waitKey(1) & 0xFF
        if text == "Moving Object Detected" and key == ord("p"):
            cv2.imwrite("screenshot.jpg", img)
            ok = True
        if key == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()

saveImage()
colorCalibrationMethod()