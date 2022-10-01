import cv2
import dlib
import numpy as np

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)

px_per_foot = 180

canvas = np.zeros((1080, 1920, 3), np.uint8)

while True:
    _, image = cap.read()
    canvas[:]=0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h,w,c = image.shape
    center_x = w/2
    center_y = h/2

    rects = detector(gray, 1)
    # Detect landmarks for each face
    for rect in rects:
        t,b,l,r = rect.top(), rect.bottom(), rect.left(), rect.right()
        x = (l+r)/2
        y = (t+b)/2
        for depth in range(1, 20):
            depth /= 2
            x_offset=int(center_x+(x-center_x)/depth)
            y_offset=int(center_y+(center_y-y)/depth)
            size=int(200//depth)

            cv2.rectangle(canvas, (x_offset-size, y_offset-size), (x_offset+size, y_offset+size), (0, int(255 - depth * 16), 0), 1)

    cv2.imshow('HeadBob', canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
