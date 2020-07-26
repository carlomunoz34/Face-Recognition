import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from siamese.SiameseNetwork import FaceRecognition
from siamese.FacesDataset import FacesDataset
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from glob import glob


def contrastive_loss(labels: torch.Tensor, predicted:torch.Tensor) -> torch.Tensor:
    """
    Loss for siamese networks
    :param labels: torch.Tensor
        The true labels of the data
    :param predicted: torch.Tensor
        The predicted labels of the data
    :return: torch.Tensor
        A tensor with just one element containing the loss
    """
    no_match = F.relu(1 - labels)  # max(margin - y, 0)
    return torch.mean(labels * predicted * predicted + (1 - predicted) * no_match * no_match)


def train(model, train_set, test_set, epochs, learning_rate, cuda=True, start_epoch=0,
          patience: int = 5) -> (list, list):
    """
    Train a siamese network with Adam optimizer and contrastive loss.
    :param model: torch.nn.Module
        The Pytorch model to train
    :param train_set: DataLoader
        The data to train the model
    :param test_set: DataLoader
        The data to test the efficiency of the model
    :param epochs: int
        The number of epochs to train
    :param learning_rate: float
        Learning rate used during the train phase
    :param cuda: bool
        If the model is in cuda
    :param start_epoch: int
        To continue the training. Indicates in what epoch the current train will start.
    :param patience: int
        The number of epochs for early stopping to end the training process
    :return: (list, list)
        The train and test losses
    """
    losses = []
    test_losses = []
    early_stopping_diff = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = contrastive_loss

    for epoch in range(start_epoch, epochs):
        sys.stdout.flush()
        epoch_loss = 0
        for first_image, second_image, labels in tqdm(train_set, total=len(train_set),
                                                      desc='Train'):
            if cuda:
                first_image = first_image.cuda()
                second_image = second_image.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            predicted = model(first_image, second_image)

            current_loss = loss(labels, predicted)
            epoch_loss += current_loss

            current_loss.backward()
            optimizer.step()

        # Test step
        sys.stdout.flush()
        test_loss = 0
        with torch.no_grad():
            for first_image, second_image, labels in tqdm(test_set, total=len(test_set),
                                                          desc="Test"):
                if cuda:
                    first_image = first_image.cuda()
                    second_image = second_image.cuda()
                    labels = labels.cuda()

                predicted = model(first_image, second_image)
                test_loss += loss(labels, predicted)

        epoch_loss /= len(train_set)
        test_loss /= len(test_set)
        losses.append(epoch_loss)
        test_losses.append(test_loss)

        print(f'Epoch: {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, ' +
              f'Test loss: {test_loss:.4f}. Saving model.')

        torch.save(model.state_dict(), f'./models/siamese_{epoch+1}.pt')

        # early stopping
        if epoch > patience:
            stop = True
            past_test_loss = test_losses[-patience]
            for test_loss in test_losses[-patience + 1:]:
                if abs(past_test_loss - test_loss) > early_stopping_diff:
                    stop = False

            if stop:
                break

    return losses, test_losses


if __name__ == '__main__':
    lr = 0.0001
    batch_size = 128
    epochs = 30
    start_epoch = 0
    model_path = glob("./models/*")[-1]

    train_data = FacesDataset(train=True, half=False)
    train_loader = DataLoader(train_data, batch_size, False)

    test = FacesDataset(train=False, half=False)
    test_loader = DataLoader(test, batch_size, False)

    siameseNetwork = FaceRecognition().cuda()

    losses, test_losses = train(siameseNetwork, train_loader, test_loader, epochs, lr,
                                start_epoch=start_epoch)

    plt.plot(losses, label='Train losses')
    plt.plot(test_losses, label='Test losses')
    plt.legend()
    plt.savefig(f"losses {start_epoch}-{epochs}.png")
