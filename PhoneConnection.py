import urllib.request
import cv2
import numpy as np
import time

url = 'http://:172.30.210.210:8080/shot.jpg'
while True:
        imgResp = urllib.request.urlopen(url)

        imgNp=np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img=cv2.imdecode(imgNp, -1)
        cv2.imshow('IPWebcam', img)

        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

