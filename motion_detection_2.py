import cv2
import os
import time

class MotionDetector:
    def __init__(self, video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.mog = cv2.createBackgroundSubtractorMOG2()
        self.snapshot_list = []

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = self.mog.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return True
        return False

    def capture_snapshots(self):
        if not os.path.exists("snapshots"):
            os.mkdir("snapshots")
        significant_motion = False

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera not initialised properly")
                break

            motion_detected = self.detect_motion(frame)

            if motion_detected:
                if not significant_motion:
                    significant_motion = True
                    time.sleep(1)  # Wait for 1 second after significant motion ends
                else:
                    ret, snapshot = self.cap.read()
                    if ret:
                        timestamp = time.strftime("%Y%m%d%H%M%S")
                        snapshot_filename = f"snapshots/snapshot_{timestamp}.jpg"
                        cv2.imwrite(snapshot_filename, snapshot)
                        self.snapshot_list.append(snapshot_filename)
            else:
                significant_motion = False
            cv2.putText(frame, f"Move Frame to capture images", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Motion Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = MotionDetector()
    try:
        detector.capture_snapshots()
    finally:
        detector.release()
