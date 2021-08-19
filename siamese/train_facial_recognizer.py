from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from siamese import FacialRecognizer
from data import FacesDataset
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import sys
import copy


def train(model: torch.nn.Module, train_set, test_set, epochs_, learning_rate, model_name: str, path_to_save: str,
          start_epoch_=0, adam: bool = True, patience: int = 5) -> (list, list):
    """
    Train a siamese network with Adam or RMSProp optimizer and contrastive loss.
    :param model: torch.nn.Module
        The Pytorch model to train
    :param train_set: DataLoader
        The data to train the model
    :param test_set: DataLoader
        The data to test the efficiency of the model
    :param epochs_: int
        The number of epochs to train. If -1, it will train until the early stopping stops the training
    :param learning_rate: float
        Starting learning rate used during the train phase
    :param model_name: str
        Specify the model name, just to save the files correctly
    :param path_to_save: str
        The path to save the model
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
    acc = []
    test_accs = []
    best_score = float("inf")
    best_model = None
    not_improved = 0
    until_converge = False

    if adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if start_epoch_ != 0:
        lr_scheduler.load_state_dict(torch.load(f"{path_to_save}/lr_scheduler_{start_epoch_ - 1}.pt"))

    if epochs_ == -1:
        epochs_ = start_epoch_ + 1
        until_converge = True

    model.train()
    loss = BCEWithLogitsLoss()

    epoch = start_epoch_
    while epoch < epochs_:
        sys.stdout.flush()
        epoch_loss = 0
        epoch_acc = 0

        for first_image, second_image, labels in tqdm(train_set, total=len(train_set), desc='Train'):
            first_image = first_image.cuda()
            second_image = second_image.cuda()
            labels = labels.cuda().unsqueeze(1).float()
            optimizer.zero_grad()

            predicted = model(first_image, second_image)

            current_loss = loss(predicted, labels)
            current_loss.backward()

            epoch_loss += current_loss.detach().cpu().item()
            predicted = torch.sigmoid(predicted)
            epoch_acc += get_number_of_corrects(labels, predicted) / len(first_image)

            optimizer.step()

        # Test step
        sys.stdout.flush()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for first_image, second_image, labels in tqdm(test_set, total=len(test_set), desc="Test"):
                first_image = first_image.cuda()
                second_image = second_image.cuda()
                labels = labels.cuda().unsqueeze(1).float()

                predicted = model(first_image, second_image)
                predicted = torch.sigmoid(predicted)
                test_loss += loss(predicted, labels)
                test_acc += get_number_of_corrects(labels, predicted) / len(first_image)

        epoch_loss /= len(train_set)
        test_loss /= len(test_set)
        losses_.append(epoch_loss)
        test_losses_.append(test_loss)

        epoch_acc /= len(train_set)
        test_acc /= len(test_set)
        acc.append(epoch_acc)
        test_accs.append(test_acc)

        torch.save(model.state_dict(), f'{path_to_save}/{model_name}_{epoch}.pt')
        torch.save(lr_scheduler.state_dict(), f'{path_to_save}/lr_scheduler_{epoch}.pt')

        # Get the best model
        if test_loss < best_score:
            best_score = test_loss
            best_model = copy.deepcopy(model.state_dict())
            not_improved = 0

        else:
            not_improved += 1

        # Early stopping
        if not_improved == patience:
            break

        if until_converge:
            epochs_ += 1
            message = f'Epoch: {epoch + 1}, ' \
                      f'Loss: {epoch_loss:.4f}, Test loss: {test_loss:.4f}, ' + \
                      f'Accuracy: {epoch_acc:.4f}, Test accuracy: {test_acc:.4f}, ' \
                      f'Epochs w/o improvement: {not_improved}\n'
        else:
            message = f'Epoch: {epoch + 1}/{epochs_}, ' \
                      f'Loss: {epoch_loss:.4f}, Test loss: {test_loss:.4f}, ' + \
                      f'Accuracy: {epoch_acc:.4f}, Test accuracy: {test_acc:.4f}, ' \
                      f'Epochs w/o improvement: {not_improved}\n'
        print(message)

        with open(f'{path_to_save}/train_{model_name}.log', 'a') as f:
            f.write(message)
            f.close()

        epoch += 1

    torch.save(best_model, f'{path_to_save}/best-{model_name}.pt')
    return losses_, test_losses_, acc, test_accs


def get_number_of_corrects(true: torch.Tensor, predicted: torch.Tensor):
    return predicted.round().eq(true).sum().item()


def train_facial_recognizer():
    lr = 0.001
    batch_size = 32
    epochs = -1
    start_epoch = 0
    path = './models/binary_cross_entropy'

    bases = ['resnet101']
    for base in bases:
        train_data = FacesDataset(train=True, base=base)
        train_loader = DataLoader(train_data, batch_size, False)

        test = FacesDataset(train=False, base=base)
        test_loader = DataLoader(test, batch_size, False)

        model = FacialRecognizer().cuda()
        model_name_ = 'facial_recognizer'

        print(f"Starting train of {base}\n")
        losses, test_losses, acc, test_acc = train(model, train_loader, test_loader, epochs, lr,
                                                   model_name_, start_epoch_=start_epoch, adam=False, patience=10,
                                                   path_to_save=path)

        plt.plot(losses, label='Train losses')
        plt.plot(test_losses, label='Test losses')
        plt.legend()
        plt.savefig(f"losses_{model_name_}.png")
        plt.clf()

        plt.plot(acc, label='Train accuracies')
        plt.plot(test_acc, label='Test accuracies')
        plt.legend()
        plt.savefig(f"accuracies_{model_name_}.png")
        plt.clf()

        # Fine tune the model
        print(f"Starting fine tune of {base}\n")
        ft_lr = lr * 0.01
        ft_epochs = -1
        ft_start_epoch = 0
        model.load(f'./models/binary_cross_entropy/best-{model_name_}.pt')
        model.prepare_for_fine_tuning()
        model_name_ += '_ft'

        losses, test_losses, acc, test_acc = train(model, train_loader, test_loader, ft_epochs, ft_lr,
                                                   model_name_, path_to_save=path, start_epoch_=ft_start_epoch,
                                                   adam=False, patience=10)

        plt.plot(losses, label='Train losses')
        plt.plot(test_losses, label='Test losses')
        plt.legend()
        plt.savefig(f"losses_{model_name_}.png")
        plt.clf()

        plt.plot(acc, label='Train accuracies')
        plt.plot(test_acc, label='Test accuracies')
        plt.legend()
        plt.savefig(f"accuracies_{model_name_}.png")
        plt.clf()


if __name__ == '__main__':
    train_facial_recognizer()
