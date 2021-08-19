import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, inception_v3, resnet34, resnet101, resnet152, densenet161, wide_resnet101_2
from data.ImageSelector import ImageSelector
import os
import numpy as np
from utils.constants import LATENT_DIM, HIDDEN_SIZE
from .predictors import LinearPredictor


class SiameseNetwork(nn.Module):
    """
    Face recognition model implemented with a siamese network with a
    MobileNetV2 in order to make it faster
    """

    def __init__(self, base: str = 'mobilenet', load_weights=True, latent_dim=4096):
        """
        Initialize the siamese network and loads MobileNet
        :param base: str
            Choose the base model between:
                'mobilenet' -> MobileNet V2,
                'inception' -> Inception V3,
                'resnet' -> ResNet 34,
                'densenet' -> DenseNet 161
        """
        super().__init__()

        self.initialized = False
        self.vectors = None
        self.is_cuda = False

        # We are going to substitute the classifier with
        # a custom ANN to make the latent space
        if base == 'mobilenet':
            self.model = mobilenet_v2(pretrained=load_weights)

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
                nn.Linear(in_features=HIDDEN_SIZE, out_features=latent_dim)
            )

        elif base == 'inception':
            self.model = inception_v3(pretrained=load_weights, aux_logits=False)

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
                nn.Linear(in_features=HIDDEN_SIZE, out_features=latent_dim)
            )

        elif base == 'resnet':
            self.model = resnet34(pretrained=load_weights)

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=512, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=latent_dim)
            )

        elif base == 'resnet101':
            self.model = resnet101(pretrained=load_weights)

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
                nn.Linear(in_features=HIDDEN_SIZE, out_features=latent_dim)
            )

        elif base == 'resnet152':
            self.model = resnet152(pretrained=load_weights)

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=2048, out_features=HIDDEN_SIZE, bias=False),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE, bias=False),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=latent_dim)
            )

        elif base == 'wide_resnet':
            self.model = wide_resnet101_2(pretrained=load_weights)

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=2048, out_features=HIDDEN_SIZE, bias=False),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE, bias=False),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=latent_dim)
            )

        elif base == 'densenet':
            self.model = densenet161(pretrained=load_weights)

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=2208, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE),
                nn.BatchNorm1d(num_features=HIDDEN_SIZE),
                nn.SELU(),
                nn.Linear(in_features=HIDDEN_SIZE, out_features=latent_dim)
            )

        else:
            raise ValueError('"base" is not a valid model')

        if not load_weights:
            for param in self.model.parameters():
                param.requires_grad = True

        self.name = f'{base}_{LATENT_DIM}'
        self.base = base

    def forward(self, first_image: torch.Tensor, second_image: torch.Tensor) \
            -> torch.Tensor:
        """
        A step through the network. Both images are passed and the output is
        the norm distance between these to outputs
        :param first_image: torch.Tensor
            The first image to process
        :param second_image: torch.Tensor
            The second image to process
        :return: torch.Tensor
            The norm of the difference between the two processed images
        """
        first_latent = self.model(first_image)
        second_latent = self.model(second_image)

        return torch.norm(first_latent - second_latent, dim=-1)

    @torch.no_grad()
    def super(self, first_image: torch.Tensor, second_image: torch.Tensor) -> torch.Tensor:
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

        predictions = torch.round(torch.reshape(distance, (-1,)))

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

    def initialize(self, predictor: LinearPredictor, cuda: bool = False, half: bool = False):
        """
        initialize the network to have a faster performance with the database
        :param half: bool
            Specify if the device is half
        :param cuda: bool
            Specify if the device is cuda
        :param predictor: Predictor
            The predictor to calculate the actual probabilities during inference
        """
        self.is_cuda = cuda
        image_selector = ImageSelector()
        faces = torch.from_numpy(image_selector.get_all_faces(pre_process=True))
        if self.is_cuda:
            faces = faces.cuda()

        if half:
            faces = faces.half()

        self.vectors = self.model(faces)
        self.initialized = True
        self.eval()
        self.predictor = predictor
        return self

    def get_database_prediction(self, face: np.ndarray) -> (int, torch.Tensor):
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

        norm = torch.norm(self.vectors - latent_vector, dim=1)
        prediction = self.predictor.predict(norm)
        prob = torch.max(prediction)
        index = torch.argmax(prediction)
        return index, prob

    def prepare_for_fine_tuning(self):
        """
        Unfreeze the last layers of the convolutional base in order to train in
        fine tuning way
        """
        if self.base == 'mobilenet':
            for parameter in self.model.features[13:].parameters():
                parameter.requires_grad = True

        elif self.base == 'inception':
            for parameter in self.model.Mixed_7a.parameters():
                parameter.requires_grad = True
            for parameter in self.model.Mixed_7b.parameters():
                parameter.requires_grad = True
            for parameter in self.model.Mixed_7c.parameters():
                parameter.requires_grad = True

        elif self.base == 'resnet':
            for parameter in self.model.layer4.parameters():
                parameter.requires_grad = True
            for parameter in self.model.layer3[18:].parameters():
                parameter.requires_grad = True

        elif self.base == 'resnet101':
            for parameter in self.model.layer4.parameters():
                parameter.requires_grad = True
            for parameter in self.model.layer3[3:].parameters():
                parameter.requires_grad = True

        elif self.base == 'densenet':
            for parameter in self.model.features.denseblock4.parameters():
                parameter.requires_grad = True

        else:
            raise ValueError('"base" is not a valid model')

    def deep_fine_tune(self, depth):
        if self.base == 'resnet101':
            if depth >= 1:
                for parameter in self.model.layer4.parameters():
                    parameter.requires_grad = True
            if depth >= 2:
                for parameter in self.model.layer3[20:].parameters():
                    parameter.requires_grad = True
            if depth >= 3:
                for parameter in self.model.layer3[17:].parameters():
                    parameter.requires_grad = True
            if depth >= 4:
                for parameter in self.model.layer3[14:].parameters():
                    parameter.requires_grad = True
            if depth >= 5:
                for parameter in self.model.layer3[11:].parameters():
                    parameter.requires_grad = True
            if depth >= 6:
                for parameter in self.model.layer3[8:].parameters():
                    parameter.requires_grad = True
            if depth >= 7:
                for parameter in self.model.layer3[5:].parameters():
                    parameter.requires_grad = True
            if depth >= 8:
                for parameter in self.model.layer3.parameters():
                    parameter.requires_grad = True

        if self.base == 'resnet152':
            if depth >= 1:
                for parameter in self.model.layer4.parameters():
                    parameter.requires_grad = True
            if depth >= 2:
                for parameter in self.model.layer3[17:].parameters():
                    parameter.requires_grad = True
            if depth >= 3:
                for parameter in self.model.layer3.parameters():
                    parameter.requires_grad = True
            if depth >= 4:
                for parameter in self.model.layer2.parameters():
                    parameter.requires_grad = True
                for parameter in self.model.layer1.parameters():
                    parameter.requires_grad = True

        if self.base == 'wide_resnet':
            if depth >= 1:
                for parameter in self.model.layer4.parameters():
                    parameter.requires_grad = True
            if depth >= 2:
                for parameter in self.model.layer3[11:].parameters():
                    parameter.requires_grad = True
            if depth >= 3:
                for parameter in self.model.layer3.parameters():
                    parameter.requires_grad = True
            if depth >= 4:
                for parameter in self.model.layer2.parameters():
                    parameter.requires_grad = True
                for parameter in self.model.layer1.parameters():
                    parameter.requires_grad = True
