import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class GestureRecognizerWrapper:
    """Обёртка для MediaPipe Gesture Recognizer в режиме LIVE_STREAM."""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: путь к файлу модели gesture_recognizer.task
        """
        self.model_path = model_path
        self._latest_result = None

        # Настройка опций распознавателя
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._result_callback
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

    def _result_callback(self, result: GestureRecognizerResult, 
                         output_image: mp.Image, timestamp_ms: int):
        """Callback, сохраняющий последний полученный результат."""
        self._latest_result = result

    def process_frame(self, frame: cv2.Mat):
        """
        Отправляет кадр в асинхронный распознаватель и возвращает текущий
        распознанный жест (если есть).

        Args:
            frame: изображение в формате BGR (numpy array)

        Returns:
            tuple (gesture_name, confidence) или (None, None), если жест не найден.
        """
        # Конвертация BGR -> RGB и создание MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Монотонно возрастающий timestamp
        timestamp_ms = int(time.time() * 1000)

        # Асинхронный вызов
        self.recognizer.recognize_async(mp_image, timestamp_ms)

        # Возвращаем последний известный результат
        if self._latest_result and self._latest_result.gestures:
            top_gesture = self._latest_result.gestures[0][0]
            return top_gesture.category_name, top_gesture.score
        return None, None

    def close(self):
        """Освобождает ресурсы распознавателя."""
        if hasattr(self, 'recognizer'):
            self.recognizer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


