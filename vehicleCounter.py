import cv2
import numpy as np
from time import sleep

widthMin = 80
heightMin = 130
widthMax = 120
heightMax = 350
offset = 6
posLine = 347
delay = 60
detect = []
cars = 0


def catchCenter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('DRONEVIDEO_720.mp4')
subtract = cv2.createBackgroundSubtractorMOG2()
frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))

while True:
    success, img = cap.read()
    if success:
        time = float(1/delay)
        sleep(time)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)
        fgmask = subtract.apply(gray)
        imgSub = subtract.apply(blur)
        dilate = cv2.dilate(fgmask, np.ones((3, 3)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.morphologyEx(dilate, cv2. MORPH_CLOSE, kernel)
        dilated = cv2.morphologyEx(dilated, cv2. MORPH_CLOSE, kernel)
        outline, h = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for(i, c) in enumerate(outline):
            (x, y, w, h) = cv2.boundingRect(c)
            validateOutline = (w >= widthMin) and (h >= heightMin) and (w <= widthMax) and (h <= heightMax)
            if not validateOutline:
                continue

            cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), 2)
            center = catchCenter(x, y, w, h)
            detect.append(center)
            cv2.circle(img, center, 4, (0, 0, 255), -1)

            for (x, y) in detect:
                if y < (posLine + offset) and y > (posLine - offset):
                    cars += 1
                    detect.remove((x, y))

        cv2.putText(img, ""+str(cars), (1120, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, "Adnan", (550, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imshow("Drone Capture", img)
        cv2.imshow("Detector", dilated)
        cv2.imshow("fgmask", fgmask)

        out.write(img)

        if cv2.waitKey(1) == 27:
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

