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
        model_path = '/home/carlo/Documentos/Proyectos/Face-Recognition' \
                     '/detector/models/res10_300x300_ssd_iter_140000.caffemodel'
        conf_file = '/home/carlo/Documentos/Proyectos/Face-Recognition' \
                    '/detector/models/deploy.prototxt.txt'
        self.face_classifier = cv2.dnn.readNetFromCaffe(conf_file, model_path)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Get a face in a photo
        :param image: np.ndarray
            The photo to extract the faces
        :return: np.ndarray
            The first face detected. If image is None, returns nan. If there is no faces
            detected, returns an empty array
        """
        return self.with_coordinates(image)[0]

    def with_coordinates(self, image: np.ndarray) -> (np.ndarray, float, float, float, float):
        """
        Get a face in a photo and its coordinates int the photo
        :param image: np.ndarray
            The image to look for faces
        :return: (np.ndarray, float, float, float, float)
            The picture of the face, the start x, start y, end x, and end y
        """
        if image is None:
            raise ValueError("Image is empty")

        # Get the image dimensions
        image_height, image_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_classifier.setInput(blob)
        detections = self.face_classifier.forward()

        # Loop over the detections and get the firs one
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3: 7]
                box *= np.array([image_width, image_height, image_width, image_height])

                start_x, start_y, end_x, end_y = box.astype(np.int32)

                # Clip the values
                end_x = min(end_x, image.shape[0])
                end_y = min(end_y, image.shape[1])

                return image[start_y: end_y, start_x: end_x], start_x, start_y, end_x, end_y

        # If there is no face detected
        return np.array([]), 0., 0., 0., 0.

    def get_all_faces(self, image: np.ndarray) -> (list, list, list, list, list):
        """
        Get all the faces and its coordinates in an image
        :param image: np.ndarray
            The image to look for faces
        :return: (list, list, list, list, list)
            A list with the faces in a np.ndarray
            A list with the start x coordinates as a floats
            A list with the start y coordinates as a floats
            A list with the end x coordinates as a floats
            A list with the end y coordinates as a floats
        """
        if image is None:
            raise ValueError("Image is empty")

        # Get the image dimensions
        image_height, image_width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        self.face_classifier.setInput(blob)
        detections = self.face_classifier.forward()

        start_xs = []
        start_ys = []
        end_xs = []
        end_ys = []
        faces = []

        # Loop over the detections and get the firs one
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3: 7]
                box *= np.array([image_width, image_height, image_width, image_height])

                start_x, start_y, end_x, end_y = box.astype(np.int32)

                # Clip the values
                end_x = min(end_x, image.shape[0])
                end_y = min(end_y, image.shape[1])

                face = image[start_y: end_y, start_x: end_x]

                start_xs.append(start_x)
                start_ys.append(start_y)
                end_xs.append(end_x)
                end_ys.append(end_y)
                faces.append(face)

        # If there is no face detected
        return faces, start_xs, start_ys, end_xs, end_ys
