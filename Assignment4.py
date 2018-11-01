import cv2
import numpy as np
import argparse
import urllib
import urllib.request

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
        cv2.imshow('circle Detection', res1) # only showing circles and red



        # Check if the user pressed ESC key
        c = cv2.waitKey(5)
        if c == 27:
            break

    cv2.destroyAllWindows()

# cv2.imshow("output", gray)
# cv2.waitKey(0)
