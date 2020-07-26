import torch
from torch.utils.data import Dataset
import cv2
from glob import glob
import numpy as np
import pandas as pd
from detector.FaceDetector import FaceDetector
from utils.constants import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS


class FacesDataset(Dataset):
    """
    Dataset that contains tha pair of images of faces
    """
    def __init__(self, train: bool = True, validation: bool = False):
        """
        Initialize the dataset
        :param train: bool
            True if is going to be the train set, false if is going
            to be the test set
        :param validation: bool
            Indicates if the data will be the validation set
        """
        dataset_path = "/home/carlo/Documentos/Datasets/CelebA/"
        self.images_path = dataset_path + "img_align_celeba/"
        files = glob(self.images_path + "*")
        self.train = train
        table = pd.read_csv(dataset_path + "identity_CelebA.csv").values
        self.images = dict()
        self.faceDetector = FaceDetector()

        for file, idx in table:
            if idx not in self.images:
                self.images[idx] = [file]
            else:
                self.images[idx].append(file)

        if train:
            self.files = files[:100000]
        elif not validation:
            self.files = files[100000: 120000]
        elif validation:
            self.files = files[120000: 140000]

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor, int):
        """
        Get two images and a label. There is 50% of probability for the image to be of
        the same person.
        :param item: int
            Ignored
        :return: (torch.Tensor, torch.Tensor, int)
            The first and second image and the label
        """
        # It will choose randomly if this sample is going to be
        # the same person or 2 different persons
        same_person = np.random.random()
        label = 0

        if same_person > 0.5:  # It is going to be the same person
            label = 1
            get = self.__get_same_person

        else:  # Two different persons
            get = self.__get_different_persons

        while True:
            first_path, second_path = get()
            first_image = cv2.imread(self.images_path + first_path)
            second_image = cv2.imread(self.images_path + second_path)

            first_face = self.faceDetector(first_image)
            second_face = self.faceDetector(second_image)

            if len(first_face) == 0 or len(second_face) == 0:
                continue

            try:
                first_face = cv2.resize(first_face, (IMG_HEIGHT, IMG_WIDTH))
                second_face = cv2.resize(second_face, (IMG_HEIGHT, IMG_WIDTH))
                break
            except:
                pass

        first_face = cv2.cvtColor(first_face, cv2.COLOR_BGR2RGB)

        second_face = cv2.cvtColor(second_face, cv2.COLOR_BGR2RGB)

        first_face = first_face.transpose((2, 0, 1)) / 127.5 - 1
        first_face = torch.from_numpy(first_face.astype(np.float32))
        first_face = first_face.squeeze()

        second_face = second_face.transpose((2, 0, 1)) / 127.5 - 1
        second_face = torch.from_numpy(second_face.astype(np.float32))
        second_face = second_face.squeeze()

        return first_face, second_face, label

    def __len__(self) -> int:
        """
        Return a fixed length for the dataset
        :return: int
            20,000 if is train set, 2,000 if is test set
        """
        return 20000 if self.train else 2000

    def __get_same_person(self) -> (str, str):
        """
        Get the paths of two different images from the same person
        :return: (str, str)
            The paths to the images
        """
        idx = 0
        while True:
            idx = np.random.choice(list(self.images.keys()))

            if len(self.images[idx]) > 1:
                break

        while True:
            first_image = np.random.choice(self.images[idx])
            second_image = np.random.choice(self.images[idx])

            if first_image != second_image:
                break

        return first_image, second_image

    def __get_different_persons(self) -> (str, str):
        """
        Get the paths of two different images from the same person
        :return: (str, str)
            The paths to the images
        """
        while True:
            first_idx = np.random.choice(list(self.images.keys()))
            second_idx = np.random.choice(list(self.images.keys()))

            if first_idx != second_idx:
                break

        first_image = np.random.choice(self.images[first_idx])
        second_image = np.random.choice(self.images[second_idx])

        return first_image, second_image
