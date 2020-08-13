import torch
from torch.utils.data import Dataset
import cv2
from glob import glob
import numpy as np
import pandas as pd
from utils.constants import MOBILENET_IMG_HEIGHT, MOBILENET_IMG_WIDTH
from utils.constants import INCEPTION_IMG_HEIGHT, INCEPTION_IMG_WIDTH
from utils.constants import RESNET_IMG_HEIGHT, RESNET_IMG_WIDTH
from utils.constants import DENSENET_IMG_HEIGHT, DENSENET_IMG_WIDTH
from data import process_image


class FaceTriplet(Dataset):
    """
    Class that provides the images in the necessary format to apply the triplet loss.
    """
    def __init__(self, base: str, train: bool = True):
        """
        Initialize the dataset
        :param base: The base of the siamese network. Just for the resizing
        :param train: bool
            True if it is the train set
        """
        super().__init__()

        dataset_path = "/home/carlo/Documentos/Datasets/CelebA/"
        self.images_path = dataset_path + "img_celeba/"
        self.train = train
        table = pd.read_csv(dataset_path + "identity_CelebA.csv").values
        boxes_table = pd.read_csv(dataset_path + "list_bbox_celeba.csv").values
        self.identities = dict()
        self.boxes = dict()

        for file, idx in table:
            if idx not in self.identities:
                self.identities[idx] = [file]
            else:
                self.identities[idx].append(file)

        for file, x1, y1, width, height in boxes_table:
            x2 = x1 + width
            y2 = y1 + height
            if file not in self.boxes:
                self.boxes[file] = [x1, y1, x2, y2]

        self.base = base

        if self.base == 'mobilenet':
            self.height = MOBILENET_IMG_HEIGHT
            self.width = MOBILENET_IMG_WIDTH

        elif self.base == 'inception':
            self.height = INCEPTION_IMG_HEIGHT
            self.width = INCEPTION_IMG_WIDTH

        elif self.base == 'resnet' or self.base == 'resnet101':
            self.height = RESNET_IMG_HEIGHT
            self.width = RESNET_IMG_WIDTH

        elif self.base == 'densenet':
            self.height = DENSENET_IMG_HEIGHT
            self.width = DENSENET_IMG_WIDTH

        else:
            raise ValueError("base in not valid")

    def __getitem__(self, item) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Return 3 images: the anchor face; the positive face, which is the same person as the anchor;
        and the negative face, which is from a different person from the anchor
        :param item: int
            Ignored, it is sampled randomly
        :return: (torch.Tensor, torch.Tensor, torch.Tensor)
            The anchor image face, the positive image face, and the negative image face
        """
        negative_file = ''
        while True:
            while True:
                anchor_name = np.random.choice(list(self.identities.keys()))

                if len(self.identities[anchor_name]) > 1:
                    if len(self.identities[anchor_name]) == 2:
                        anchor_file = self.identities[anchor_name][0]
                        positive_file = self.identities[anchor_name][1]

                    else:
                        anchor_file, positive_file = np.random.choice(self.identities[anchor_name], 2)
                        if anchor_file == positive_file:
                            continue

                    negative_name = np.random.choice(list(self.identities.keys()))
                    if negative_name == anchor_name:
                        continue

                    negative_file = np.random.choice(self.identities[negative_name])
                    break

            anchor_image = cv2.imread(self.images_path + anchor_file).astype(np.float32)
            positive_image = cv2.imread(self.images_path + positive_file).astype(np.float32)
            negative_image = cv2.imread(self.images_path + negative_file).astype(np.float32)

            anchor_x1, anchor_y1, anchor_x2, anchor_y2 = self.boxes[anchor_file]
            positive_x1, positive_y1, positive_x2, positive_y2 = self.boxes[positive_file]
            negative_x1, negative_y1, negative_x2, negative_y2 = self.boxes[negative_file]

            anchor_image = anchor_image[anchor_y1: anchor_y2, anchor_x1: anchor_x2]
            positive_image = positive_image[positive_y1: positive_y2, positive_x1: positive_x2]
            negative_image = negative_image[negative_y1: negative_y2, negative_x1: negative_x2]

            try:
                anchor_image = cv2.resize(anchor_image, (self.width, self.height))
                positive_image = cv2.resize(positive_image, (self.width, self.height))
                negative_image = cv2.resize(negative_image, (self.width, self.height))
            except:
                continue

            anchor_image = process_image(anchor_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            positive_image = process_image(positive_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            negative_image = process_image(negative_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            anchor_image = torch.from_numpy(anchor_image)
            positive_image = torch.from_numpy(positive_image)
            negative_image = torch.from_numpy(negative_image)

            return anchor_image, positive_image, negative_image

    def __len__(self):
        """
        Return a fixed length for the dataset
        :return: int
            20,000 if is train set, 2,000 if is test set
        """
        return 20000 if self.train else 2000
