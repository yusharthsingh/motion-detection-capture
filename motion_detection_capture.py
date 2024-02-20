import cv2
import numpy as np
import time
import os
# Create the "snapshots" folder if it doesn't exist
cap = cv2.VideoCapture(0)
mog = cv2.createBackgroundSubtractorMOG2()
motion_detected = False
snapshot_list = []


def motion_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = mog.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True
        return motion_detected

def image_collection():
    if not os.path.exists("snapshots"):
        os.mkdir("snapshots")
    significant_motion = False

    while True:
        ret, frame = cap.read()
        motion_detected = motion_detection(frame)
        if motion_detected:
            if not significant_motion:
                significant_motion = True
                time.sleep(1)  # Wait for 1 second after significant motion ends
            else:
                ret, snapshot = cap.read()
                if ret:
                    timestamp = time.strftime("%Y%m%d%H%M%S")
                    snapshot_filename = f"snapshots/snapshot_{timestamp}.jpg"
                    cv2.imwrite(snapshot_filename, snapshot)
                    snapshot_list.append(snapshot_filename)
                motion_detected = False
        else:
            significant_motion = False
        cv2.imshow('Motion Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_collection()