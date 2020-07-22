import cv2
import numpy as np


class FaceDetector:
    """
    A face detector class. It detects faces in photos represented in
    a numpy's array
    """
    def __init__(self):
        """
        Initialize the model
        """
        haarcascade_path = 'Haarcascades/haarcascade_frontalface_default.xml'
        self.face_classifier = cv2.CascadeClassifier(haarcascade_path)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Get a face in a photo
        :param image: np.ndarray
            The photo to extract the faces
        :return: np.ndarray
            The first face detected. If image is None, returns nan
        """

        if image is None:
            return np.nan

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        coordinates = self.face_classifier.detectMultiScale(gray, 1.3, 5)[0]
        start_x, start_y, width, height = coordinates
        end_x = start_x + width
        end_y = start_y + height

        # Clip the values
        end_x = min(end_x, image.shape[0])
        end_y = min(end_y, image.shape[1])

        return image[start_x: end_x, start_y: end_y]
