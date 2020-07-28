import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from siamese import FaceRecognition
from data import FacesDataset
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


def contrastive_loss(labels: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
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


def train(model, train_set, test_set, epochs_, learning_rate, cuda=True, start_epoch_=0,
          patience: int = 5, adam: bool = True) -> (list, list):
    """
    Train a siamese network with Adam or RMSProp optimizer and contrastive loss.
    :param model: torch.nn.Module
        The Pytorch model to train
    :param train_set: DataLoader
        The data to train the model
    :param test_set: DataLoader
        The data to test the efficiency of the model
    :param epochs_: int
        The number of epochs to train
    :param learning_rate: float
        Learning rate used during the train phase
    :param cuda: bool
        If the model is in cuda
    :param start_epoch_: int
        To continue the training. Indicates in what epoch the current train will start.
    :param patience: int
        The number of epochs for early stopping to end the training process
    :param adam: bool
        If true, uses Adam as optimizer, if false uses RMSProp
    :return: (list, list)
        The train and test losses
    """
    losses_ = []
    test_losses_ = []
    best_score = float("inf")
    best_model = None
    no_improvement = 0

    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    loss = contrastive_loss
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epoch = start_epoch
    while epoch < start_epoch + epochs:
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
        losses_.append(epoch_loss)
        test_losses_.append(test_loss)

        message = f'Epoch: {epoch + 1}/{epochs_}, Loss: {epoch_loss:.4f}, ' + \
                  f'Test loss: {test_loss:.4f}. Saving model.\n'
        sys.stdout.write(message)

        with open('./train.log', 'a') as f:
            f.write(message)

        torch.save(model.state_dict(), f'./models/{model.name}_{epoch}.pt')
        torch.save(lr_scheduler.state_dict(), f'./models/lr_scheduler_{epoch}.pt')

        # early stopping
        if test_loss < best_score:
            no_improvement = 0
            best_score = test_loss
            best_model = model.state_dict()
        else:
            no_improvement += 1

        if no_improvement == patience:
            diff = 10 - (epoch % 10)
            epoch += diff
            for j in range(diff):
                lr_scheduler.step()

        else:
            lr_scheduler.step()
            epoch += 1

    torch.save(best_model, './models/best-' + model.name)


    return losses_, test_losses_


if __name__ == '__main__':
    lr = 0.001
    batch_size = 128
    epochs = 30
    start_epoch = 0
    # model_to_load = "./models/model_3x512_300.pt"

    train_data = FacesDataset(train=True, validation=False)
    train_loader = DataLoader(train_data, batch_size, False)

    test = FacesDataset(train=False, validation=False)
    test_loader = DataLoader(test, batch_size, False)

    siameseNetwork = FaceRecognition(deep_model=True)\
        .cuda()

    losses, test_losses = train(siameseNetwork, train_loader, test_loader, epochs, lr,
                                start_epoch_=start_epoch, adam=False)

    plt.plot(losses, label='Train losses')
    plt.plot(test_losses, label='Test losses')
    plt.legend()
    plt.savefig(f"losses {siameseNetwork.name}-{epochs}.png")
