import torch
import torch.nn as nn
from data.ImageSelector import ImageSelector
import os
import numpy as np
from torchvision import models


class SiameseNetwork(nn.Module):
    """
    Face recognition model implemented with a siamese network with
    Efficientnet B3
    """

    def __init__(self, device, model_path):
        super().__init__()
        self.device = device

        latent_dim = 4096
        base_model = models.efficientnet_b3().to(device)
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1536, out_features=latent_dim, bias=True)
        ).to(device)

        self.base_model = base_model
        self.base_model.load_state_dict(torch.load(model_path))
        self.predictor = nn.Linear(in_features=1, out_features=1).to(device)

        self.vectors = None
        self.initialized = False

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
        image_1_features = self.base_model_model(first_image)
        image_2_features = self.base_model(second_image)

        dist = torch.linalg.norm(image_1_features - image_2_features, 2, dim=-1)

        return self.predictor(dist)

    @torch.no_grad()
    def predict_proba(self, first_image: torch.Tensor, second_image: torch.Tensor) -> torch.Tensor:
        """
        Returns the probability that the person in the first image is the same person in the second image
        :param first_image: torch.Tensor
            A batch with the first images
        :param second_image: torch.Tensor
            A batch with the second images
        :return: torch.Tensor
            The probabilities for each instance
        """
        predictions = self(first_image, second_image)
        return torch.sigmoid(predictions)

    @torch.no_grad()
    def predict(self, first_image: torch.Tensor, second_image: torch.Tensor,
                threshold: float = 0.8) -> torch.Tensor:
        """
        Predict if the two images belongs to the same person
        :param first_image: torch.Tensor
            A batch with the first images
        :param second_image: torch.Tensor
            A batch with the second images
        :param threshold: float
            If the probability of an image is equal or higher to this threshold
            then, the output for that instance is going to be 1, else 0
        :return: torch.Tensor
            The prediction for each instance
        """

        probabilities = self.predict_proba(first_image, second_image)
        return (probabilities >= threshold).float()

    def load(self, path):
        """
        Loads a pretrained siamese model
        :param path: str
            The path that contains the pretrained model file
        :return: FaceRecognition
            The model with the weights loaded
        """
        if not os.path.isfile(path):
            raise IOError(f"{path} is not a valid path!")
        self.load_state_dict(torch.load(path))
        return self

    def initialize(self):
        """
        initialize the network to have a faster performance with the database
        """
        image_selector = ImageSelector()
        faces = torch.from_numpy(image_selector.get_all_faces(pre_process=True))

        faces = faces.to(self.device)

        self.vectors = self.base_model(faces)
        self.eval()
        self.initialized = True
        return self

    def get_database_prediction(self, face: np.ndarray) -> (int, torch.Tensor):
        """
        Predict if the passes face belong to someone in the database
        :param face: np.ndarray
            Tha face to recognize
        :return: int
            The index of the person who belongs the face, or -1 if it didn't find a match
        """
        assert self.initialized
        face = torch.from_numpy(face).to(self.device)

        latent_vector = self.base_model(face)

        norm = torch.norm(self.vectors - latent_vector, 2, dim=1)
        prediction = self.predictor(norm)

        prob = torch.max(prediction)
        index = torch.argmax(prediction)
        return index, prob
