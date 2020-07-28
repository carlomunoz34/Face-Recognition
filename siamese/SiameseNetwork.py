import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, inception_v3, resnet34
from data.ImageSelector import ImageSelector
import os
import numpy as np
from utils.constants import LATENT_DIM, HIDDEN_SIZE


class FaceRecognition(nn.Module):
    """
    Face recognition model implemented with a siamese network with a
    MobileNetV2 in order to make it faster
    """

    def __init__(self, base: str = 'mobilenet'):
        """
        Initialize the siamese network and loads MobileNet
        :param base: str
            Choose the base model between:
                'mobilenet' -> MobileNet V2,
                'inception' -> Inception V3,
                'resnet' -> ResNet 34
        """
        super().__init__()

        self.initialized = False
        self.vectors = None
        self.is_cuda = False

        # We are going to substitute the classifier with
        # a custom ANN to make the latent space
        if base == 'mobilenet':
            self.model = mobilenet_v2(pretrained=True)

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=1280, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=LATENT_DIM)
            )

        elif base == 'inception':
            self.model = inception_v3(pretrained=True)

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=2048, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=LATENT_DIM)
            )

        elif base == 'resnet':
            self.model = resnet34(pretrained=True)

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=LATENT_DIM)
            )

        else:
            raise ValueError('"base" is not a valid model')

        self.name = f'{base}_{LATENT_DIM}'

    def forward(self, first_image: torch.Tensor, second_image: torch.Tensor) \
            -> torch.Tensor:
        """
        A step through the network. Both images are passed and the output is
        the norm distance between these to outputs
        :param first_image:
        :param second_image:
        :return:
        """
        first_latent = self.model(first_image)
        second_latent = self.model(second_image)

        return torch.norm(first_latent - second_latent, dim=-1)

    @torch.no_grad()
    def predict(self, first_image: torch.Tensor, second_image: torch.Tensor) -> torch.Tensor:
        """
        Predict if the two images belongs to the same person
        :param first_image: torch.Tensor
            A batch with the first images
        :param second_image: torch.Tensor
            A batch with the second images
        :return: torch.Tensor
            The probabilities for each instance
        """
        distance = self(first_image, second_image)

        predictions = torch.round(torch.reshape(distance, (-1,)) - 0.2)

        # if 0, the images are from the same person. We need to change it to 1
        predictions = (predictions == 0).int()

        return predictions

    def load(self, path):
        """
        Loads a pre trained siamese model
        :param path: str
            The path that contains the pre trained model file
        :return: FaceRecognition
            The model with the weights loaded
        """
        if not os.path.isfile(path):
            raise IOError(f"{path} is not a valid path!")
        self.load_state_dict(torch.load(path))
        return self

    def initialize(self, cuda: bool = False):
        """
        initialize the network to have a faster performance with the database
        :param cuda: bool
            Specify if the device is cuda
        :return:
        """
        self.is_cuda = cuda
        image_selector = ImageSelector()
        faces = torch.from_numpy(image_selector.get_all_faces(pre_process=True))
        if self.is_cuda:
            faces = faces.cuda()

        self.vectors = self.model(faces)
        self.initialized = True
        return self

    def get_database_prediction(self, face: np.ndarray) -> int:
        """
        Predict if the passes face belong to some one in the database
        :param face: np.ndarray
            Tha face to recognize
        :return: int
            The index of the person who belongs the face, or -1 if it didn't find a match
        """
        assert self.initialized
        face = torch.from_numpy(face)
        if self.is_cuda:
            face = face.cuda()

        latent_vector = self.model(face)

        idx, prediction = torch.max(torch.norm(self.vectors - latent_vector), 0)
        if prediction > 0.70:
            return idx.item()

        return -1
