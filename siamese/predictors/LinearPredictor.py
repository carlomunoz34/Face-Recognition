import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class LinearPredictor(nn.Module):
    """
    Auxiliary module that helps the siamese module to predict if
    to pictures are from the same person.
    It takes the norm of the difference of the 2 outputs of the siamese network as input.
    """
    def __init__(self):
        """
        Initialize the module as just a simple logistic regression.
        """
        super().__init__()
        self.out = nn.Linear(in_features=1, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        A pass forward the model. To predict a probability, use predict
        :param X: torch.Tensor
            The difference of the 2 outputs of the siamese network
        :return: torch.Tensor
            Output of a the linear layer
        """
        if len(X.shape) == 1:
            X = X.unsqueeze(1)
        return self.out(X)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict if the difference of the two vectors are from the same person
        :param X: torch.Tensor
            The difference of the 2 outputs of the siamese network
        :return: torch.Tensor
            The probability of the to pictures belong to the same person
        """
        return torch.sigmoid(self(X))

    def load(self, path):
        """
        Loads a pre trained model
        :param path: str
            The path that contains the pre trained model file
        """
        if not os.path.isfile(path):
            raise IOError(f"{path} is not a valid path!")
        self.load_state_dict(torch.load(path))


class MLPPredictor(nn.Module):
    """
    Auxiliary module that helps the siamese module to predict if
    to pictures are from the same person.
    It takes the norm of the difference of the 2 outputs of the siamese network as input.
    """
    def __init__(self):
        """
        Initialize the module as just a simple logistic regression.
        """
        super().__init__()
        self.hidden = nn.Linear(in_features=LATENT_DIM, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        A pass forward the model. To predict a probability, use predict
        :param X: torch.Tensor
            The difference of the 2 outputs of the siamese network
        :return: torch.Tensor
            Output of a the linear layer
        """
        return self.out(F.elu(self.hidden(X)))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict if the norm of the two vectors are from the same person
        :param X: torch.Tensor
            The difference of the 2 outputs of the siamese network
        :return: torch.Tensor
            The probability of the to pictures belong to the same person
        """
        return torch.sigmoid(self(X))

    def load(self, path):
        """
        Loads a pre trained model
        :param path: str
            The path that contains the pre trained model file
        """
        if not os.path.isfile(path):
            raise IOError(f"{path} is not a valid path!")
        self.load_state_dict(torch.load(path))
