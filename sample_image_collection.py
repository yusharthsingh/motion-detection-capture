import cv2
import time

mog = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture(0)
def capture_images():
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prompt = f"Rotate the object and press 'c' to click 'q' to quit ({count})"
        cv2.putText(frame,prompt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("frame", frame)
        image_name = f"{count}{time.time()}.jpg"
        key = cv2.waitKey(1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = mog.apply(gray)
        # Apply morphological operations to reduce noise and fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Ignore small contours
            if cv2.contourArea(contour) < 1000:
                continue

            # Draw bounding box around contour
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("motion mask", fgmask)

        if key == ord('c'):
            count += 1
            cv2.imwrite(image_name, frame)

        elif key == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images()
