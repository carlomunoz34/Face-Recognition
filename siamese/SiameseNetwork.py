import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


LATENT_DIM = 300


class FaceRecognition(nn.Module):
    """
    Face recognition model implemented with a siamese network with a
    MobileNetV2 in order to make it faster
    """
    def __init__(self):
        """
        Initialize the siamese network and loads MobileNet
        """
        super().__init__()

        self.mobileNet = mobilenet_v2(pretrained=True)

        # To not update the mobileNet weights during training
        for parameter in self.mobileNet.parameters():
            parameter.requires_grad = False

        # We are going to substitute the classifier with
        # a dropout and a linear layer to make the latent space
        self.mobileNet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=LATENT_DIM)
        )

    def forward(self, first_image: torch.Tensor, second_image: torch.Tensor)\
            -> torch.Tensor:
        """
        A step through the network. Both images are passed and the output is
        the norm distance between these to outputs
        :param first_image:
        :param second_image:
        :return:
        """
        first_latent = self.mobileNet(first_image)
        second_latent = self.mobileNet(second_image)

        return torch.norm(first_latent - second_latent, dim=-1)

    @torch.no_grad()
    def predict(self, first_image: torch.Tensor, second_image: torch.Tensor,
                threshold: float) -> torch.Tensor:
        distance = self(first_image, second_image)

        return torch.reshape(distance, (-1,))
