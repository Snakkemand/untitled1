import imutils
import numpy as np
import argparse
import urllib
import urllib.request
import sys

import cv2
import socket

host = ''
port = 50000
backlog = 5
size = 1024
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host, port))
sock.listen(backlog)

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"


        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = ""

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


url = 'http://192.168.43.1:8080/shot.jpg'


def get_frame(scaling_factor):
    img_resp = urllib.request.urlopen(url)
    img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    # Capture the frame from video capture object
    frame = cv2.imdecode(img_arr, -1)

    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame


if __name__ == '__main__':
    scaling_factor = 0.5

    # Iterate until the user presses ESC key
    while True:
        frame = get_frame(scaling_factor)

        # Convert the HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # actual threshold for red
        lower_red = np.array([120, 100, 90])
        upper_red = np.array([220, 255, 255])

        # Threshold the HSV image to get only red color
        maskred = cv2.inRange(hsv, lower_red, upper_red)

        # actual threshold for blue
        lower_blue = np.array([80, 100, 0])
        upper_blue = np.array([200, 255, 255])

        # threshold the HSV image to get only blue
        maskblue = cv2.inRange(hsv, lower_blue, upper_blue)

        # actual threshold for green

        lower_green = np.array([80, 50, 20])
        upper_green = np.array([160, 255, 200])

        # threshold the HSV image to get only green
        maskgreen = cv2.inRange(hsv, lower_green, upper_green)

        # Bitwise-AND mask and original images (Red, Blue & Green)
        res1 = cv2.bitwise_and(frame, frame, mask=maskred)
        res2 = cv2.bitwise_and(frame, frame, mask=maskblue)
        res3 = cv2.bitwise_and(frame, frame, mask=maskgreen)
        res1 = cv2.medianBlur(res1, 5)
        res2 = cv2.medianBlur(res2, 5)
        res3 = cv2.medianBlur(res3, 5)

        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.imread("shapes_and_colors.jpg")
        resized = imutils.resize(res2, width=300)
        ratio = res2.shape[0] / float(resized.shape[0])

        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        sd = ShapeDetector()

        for c in cnts:
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int((M["m10"] / M["m00"]) * ratio)
                cY = int((M["m01"] / M["m00"]) * ratio)
            else:
                cX, cY = 0, 0
            shape = sd.detect(c)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            cv2.drawContours(res2, [c], -1, (0, 255, 0), 2)
            # cv2.putText(res2, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(res2, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(res2, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            print(shape, end='')
            # print(cX - 20, cY - 20)

            height = np.size(frame, 0)
            width = np.size(frame, 1)

            x_diff = width - (cX - 20)
            y_diff = height - (cY - 20)

            x_ratio = round(width / x_diff, 3)
            y_ratio = round(height / y_diff, 3)

            #print(" + x_ratio: ", end='')
            #print(x_ratio, end='')
            #print(" + y_ratio: ", end='')
            #print(y_ratio)

            print('Ready for connection')
            try:
                client, address = sock.accept()
                print('Client connected : ', address)
                client.send(bytes(str(shape), 'utf-8'))
            except:
                print('Something went wrong')
            sock.close()

        # detect circles in the image red
        circles = cv2.HoughCircles(maskred, cv2.HOUGH_GRADIENT, 4, 100)

        # ensure at least some circles were found
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(res1, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(res1, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image and colors
        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector1', res2)
        # cv2.imshow('circle Detection', res1)  # only showing circles and red

        # Check if the user pressed ESC key
        c = cv2.waitKey(5)
        if c == 27:
            break

    cv2.destroyAllWindows()

# cv2.imshow("output", gray)
# cv2.waitKey(0)
