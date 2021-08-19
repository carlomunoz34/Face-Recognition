import torch
import torch.nn as nn
from . import SiameseNetwork
from .predictors import LinearPredictor
import os


class FacialRecognizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.siamese_network = SiameseNetwork(base='resnet101')
        self.predictor = LinearPredictor()

    def forward(self, first_face, second_face):
        norm = self.siamese_network(first_face, second_face)
        return self.predictor(norm)

    def load(self, path):
        """
        Loads a pre trained model
        :param path: str
            The path that contains the pre trained model file
        """
        if not os.path.isfile(path):
            raise IOError(f"{path} is not a valid path!")
        self.load_state_dict(torch.load(path))

    def prepare_for_fine_tuning(self):
        self.siamese_network.prepare_for_fine_tuning()
