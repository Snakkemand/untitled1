import cv2
import numpy as np
import requests
import urllib.request

url = 'http://192.168.43.113:8888/shot.jpg'

while True:

        img_resp = urllib.request.urlopen(url)
        img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        cv2.imshow('IPWebcam', img)

        # time.sleep(0.1)
        if cv2.waitKey(1) == 27:
            break

