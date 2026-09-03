import cv2
import mediapipe as mp
from gesture_recognition import GestureRecognizerWrapper
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру.")
        return

    model_path = 'models/gesture_recognizer.task'

    with GestureRecognizerWrapper(model_path) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Пропускаем пустой кадр.")
                continue

            frame = cv2.flip(frame, 1)
            gesture_name, confidence = recognizer.process_frame(frame)
            if gesture_name is not None:
                text = f"Gesture: {gesture_name} ({confidence:.2f})"
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)


            cv2.imshow('MediaPipe Gesture Classifier', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()