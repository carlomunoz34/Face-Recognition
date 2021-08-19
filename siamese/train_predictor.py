import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from siamese import SiameseNetwork
from siamese.predictors import LinearPredictor, MLPPredictor
from data import FacesDataset
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import os
from utils.constants import LATENT_DIM


def train(model, predictor, train_set, test_set, epochs_, learning_rate, model_name: str, cuda=True, start_epoch_=0,
          adam: bool = True, patience: int = 5) -> (list, list):
    """
    Train a siamese network with Adam or RMSProp optimizer and contrastive loss.
    :param model: torch.nn.Module
        The Siamese network
    :param predictor: torch.nn.Module
        The predictor to train
    :param train_set: DataLoader
        The data to train the model
    :param test_set: DataLoader
        The data to test the efficiency of the model
    :param epochs_: int
        The number of epochs to train
    :param learning_rate: float
        Learning rate used during the train phase
    :param model_name> str
        Specify the model name, just to save the files correctly
    :param cuda: bool
        If the model is in cuda
    :param start_epoch_: int
        To continue the training. Indicates in what epoch the current train will start.
    :param adam: bool
        If true, uses Adam as optimizer, if false uses RMSProp
    :param patience: int
        The number of epochs used in the early stopping
    :return: (list, list)
        The train and test losses
    """
    losses_ = []
    test_losses_ = []
    accuracies = []
    test_accuracies = []
    best_score = float("inf")
    best_model = None
    not_improved = 0

    if adam:
        optimizer = torch.optim.Adam(predictor.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.RMSprop(predictor.parameters(), lr=learning_rate)
    loss = nn.BCEWithLogitsLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if start_epoch_ != 0:
        lr_scheduler.load_state_dict(torch.load(f"./models/lr_scheduler_{start_epoch_ - 1}.pt"))

    for epoch in range(start_epoch_, epochs):
        sys.stdout.flush()
        epoch_loss = 0
        epoch_accuracy = 0

        for first_image, second_image, labels in tqdm(train_set, total=len(train_set),
                                                      desc='Train'):
            if cuda:
                first_image = first_image.cuda()
                second_image = second_image.cuda()
                labels = labels.cuda()
            labels = labels.unsqueeze(1).float()
            optimizer.zero_grad()

            vector_norm = model(first_image, second_image).unsqueeze(1).detach()
            predicted = predictor(vector_norm)

            current_loss = loss(predicted, labels)
            epoch_loss += current_loss.detach().cpu().item()
            epoch_accuracy += get_num_of_corrects(labels, torch.sigmoid(predicted)) / len(first_image)

            current_loss.backward()
            optimizer.step()

        # Test step
        sys.stdout.flush()
        test_loss = 0
        test_accuracy = 0
        with torch.no_grad():
            for first_image, second_image, labels in tqdm(test_set, total=len(test_set),
                                                          desc="Test"):
                if cuda:
                    first_image = first_image.cuda()
                    second_image = second_image.cuda()
                    labels = labels.cuda()
                labels = labels.unsqueeze(1).float()

                vector_norm = model(first_image, second_image).unsqueeze(1).detach()
                predicted = predictor(vector_norm)

                test_loss += loss(predicted, labels).detach().cpu().item()
                test_accuracy += get_num_of_corrects(labels, torch.sigmoid(predicted)) / len(first_image)

        epoch_loss /= len(train_set)
        test_loss /= len(test_set)

        losses_.append(epoch_loss)
        test_losses_.append(test_loss)

        epoch_accuracy /= len(train_set)
        test_accuracy /= len(test_set)

        accuracies.append(epoch_accuracy)
        test_accuracies.append(test_accuracy)

        message = f'Epoch: {epoch + 1}/{epochs_}, ' \
                  f'Loss: {epoch_loss:.4f}, Test loss: {test_loss:.4f}, ' + \
                  f'Accuracy: {epoch_accuracy}, Test Accuracy: {test_accuracy:.4f}\n'
        sys.stdout.write(message)

        with open(f'./train_{model_name}.log', 'a') as f:
            f.write(message)
            f.close()

        torch.save(predictor.state_dict(), f'./models/{model_name}_{epoch}.pt')
        torch.save(lr_scheduler.state_dict(), f'./models/lr_scheduler_{epoch}.pt')

        # Get the best model
        if test_loss < best_score:
            best_score = test_loss
            best_model = predictor.state_dict()
            not_improved = 0

        else:
            not_improved += 1

        # Early stopping
        if not_improved == patience:
            break

    torch.save(best_model, './models/best-' + model_name + '.pt')

    return losses_, test_losses_, accuracies, test_accuracies


def get_num_of_corrects(true_labels: torch.Tensor, predictions: torch.Tensor) -> int:
    """
    Get the number of correct in a batch given the prediction and the true labels
    :param true_labels: torch.Tensor
        A tensor that contain the true labels
    :param predictions: torch.Tensor
        The predicted results
    :return: int
        The number of correct predictions
    """
    return predictions.round().eq(true_labels).sum().item()


if __name__ == '__main__':
    lr = 0.001
    batch_size = 32
    epochs = 100
    start_epoch = 0
    latent_dim = 4096

    train_data = FacesDataset(train=True, validation=False, base='resnet101')
    train_loader = DataLoader(train_data, batch_size, False, num_workers=8)

    test = FacesDataset(train=False, validation=False, base='resnet101')
    test_loader = DataLoader(test, batch_size, False, num_workers=14)

    models = ['./models/resnet152 2.pt', './models/resnet152 3.pt']

    for trained_model in models:
        siamese_network = SiameseNetwork(base='resnet152', latent_dim=latent_dim).cuda()
        siamese_network.load(trained_model)
        siamese_network.eval()

        predictor_ = LinearPredictor().cuda()
        model_name = 'predictor-' + trained_model.split(os.sep)[-1]

        sys.stdout.write('Training Linear predictor:\n')
        losses, test_losses, train_accuracies_, test_accuracies_ = train(
            siamese_network, predictor_, train_loader, test_loader, epochs, lr, model_name,
            start_epoch_=start_epoch, adam=False, patience=5)

        plt.plot(losses, label='Train losses')
        plt.plot(test_losses, label='Test losses')
        plt.legend()
        plt.savefig(f"losses_linear_predictor {model_name}.png")
        plt.clf()

        plt.plot(train_accuracies_, label='Train acc')
        plt.plot(test_accuracies_, label='Test acc')
        plt.legend()
        plt.savefig(f"accuracies_linear_predictor {model_name}.png")
        plt.clf()
