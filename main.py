import cv2
import mediapipe as mp

# webcam init
webcam = cv2.VideoCapture(0)

# main loop
while webcam.isOpened():
    sucess, frame = webcam.read()
    if not sucess:
        print("")
        break

    cv2.imshow("Hands", frame)

    # close condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# free cam and closed window
webcam.release()
cv2.destroyAllWindows()