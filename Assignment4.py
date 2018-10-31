import cv2
import numpy as np
import argparse


# Capture the input frame from webcam
def get_frame(cap, scaling_factor):
    # Capture the frame from video capture object
    ret, frame = cap.read()

    # Resize the input frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame


image = cv2.imread("shape_3.jpg")
output = image.copy()

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    scaling_factor = 0.5

    # Iterate until the user presses ESC key
    while True:
        frame = get_frame(cap, scaling_factor)

        # Convert the HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([120, 100, 90])
        upper_red = np.array([220, 255, 255])

        # Threshold the HSV image to get only blue color
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        res = cv2.medianBlur(res, 5)


        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect circles in the image
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 4, 100)

        # ensure at least some circles were found
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                # draw the circle in the output image, then draw a rectangle
                # corresponding to the center of the circle
                cv2.circle(res, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(res, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        cv2.imshow('Original image', frame)
        cv2.imshow('Color Detector', res)

        # Check if the user pressed ESC key
        c = cv2.waitKey(5)
        if c == 27:
            break

    cv2.destroyAllWindows()

# cv2.imshow("output", gray)
# cv2.waitKey(0)
