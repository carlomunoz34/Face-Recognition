import cv2
from detector.FaceDetector import FaceDetector
import pandas as pd
import numpy as np
from utils.constants import MOBILENET_IMG_HEIGHT, MOBILENET_IMG_WIDTH, IMG_CHANNELS


def process_image(image: np.ndarray, mean: list, std: list) -> np.ndarray:
    """
    Sort the dimensions to fit in the convolutional network
    :param image: np.ndarray
        The image to process
    :param mean: list
        List with the means to normalize the image
    :param std: list
        List with the standard deviation to normalize the image
    :return: np.ndarray
        The image processed
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.
    # normalize
    for channel in range(IMG_CHANNELS):
        image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    return (image.transpose((2, 0, 1))).astype(np.float32)


class ImageSelector:
    """
    This class return images and names given an id. This is suppose to be a
    API to manage the data and information in the database
    """
    def __init__(self):
        """
        Initialize the instance.
        """
        self.face_detector = FaceDetector()
        self.df = pd.read_csv('../data/persons.csv')

    def get_image_by_index(self, index: int) -> np.ndarray:
        """
        Get a certain image by its index.
        :param index: int
            Index of the image
        :return: np.ndarray
            The image of that person
        """
        image_path = self.df['file'][index]
        return cv2.imread("../data/" + image_path)

    def get_face_by_index(self, index: int) -> np.ndarray:
        """
        Use the face detector API to get a face of an image by its index
        :param index: int
            Index of the image
        :return: np.ndarray
            The face in that image
        """
        image = self.get_image_by_index(index)
        return cv2.resize(self.face_detector(image), (MOBILENET_IMG_HEIGHT, MOBILENET_IMG_WIDTH))

    def get_name_by_index(self, index) -> str:
        """
        Returns the name of a person by its index
        :param index: int
            The index of the person
        :return: str
            The name of the person
        """
        return self.df['name'][index]

    def get_all_faces(self, pre_process: bool = False) -> np.ndarray:
        """
        Get all the faces of the database
        :param pre_process: bool
            True if the image is pre processed to fit in the network
        :return: np.ndarray
            An (N x IMG_HEIGHT x IMG_WIDTH x IMG_CHANNELS) numpy array
            containing all the faces of the dataset
        """
        dataset_length = len(self.df)

        if pre_process:
            faces = np.zeros((dataset_length, IMG_CHANNELS, MOBILENET_IMG_HEIGHT, MOBILENET_IMG_WIDTH), dtype=np.float32)
        else:
            faces = np.zeros((dataset_length, MOBILENET_IMG_HEIGHT, MOBILENET_IMG_WIDTH, IMG_CHANNELS))

        for i in range(dataset_length):
            face = self.get_face_by_index(i)

            if pre_process:
                faces[i] = process_image(face)
            else:
                faces[i] = face

        return faces
