import torch
from tqdm import tqdm
import sys
import copy


def train(model: torch.nn.Module, train_set, test_set, epochs_, step, learning_rate, model_name: str, path_to_save: str,
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
    :param step: function
        A function that receives a data sample from the dataset and returns the loss
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

    epoch = start_epoch_
    while epoch < epochs_:
        sys.stdout.flush()
        epoch_loss = 0

        for sample in tqdm(train_set, total=len(train_set), desc='Train'):
            optimizer.zero_grad()

            current_loss = step(model, *sample)
            current_loss.backward()

            epoch_loss += current_loss.detach().cpu().item()

            optimizer.step()

        # Test step
        sys.stdout.flush()
        test_loss = 0
        with torch.no_grad():
            for sample in tqdm(test_set, total=len(test_set), desc="Test"):
                test_loss += step(model, *sample)

        epoch_loss /= len(train_set)
        test_loss /= len(test_set)
        losses_.append(epoch_loss)
        test_losses_.append(test_loss)

        torch.save(model.state_dict(), f'{path_to_save}/{model_name}_{epoch + 1}.pt')
        torch.save(lr_scheduler.state_dict(), f'{path_to_save}/lr_scheduler_{epoch + 1}.pt')

        # Get the best model
        if test_loss < best_score:
            best_score = test_loss
            best_model = copy.deepcopy(model.state_dict())
            not_improved = 0

        else:
            not_improved += 1

        if until_converge:
            epochs_ += 1
            message = f'Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, ' + \
                      f'Test loss: {test_loss:.4f}, Epochs w/o improvement: {not_improved}\n'
            full_message = f'Epoch: {epoch + 1}, Loss: {epoch_loss}, ' + \
                           f'Test loss: {test_loss}, Epochs w/o improvement: {not_improved}\n'
        else:
            message = f'Epoch: {epoch + 1}/{epochs_}, Loss: {epoch_loss:.4f}, ' + \
                      f'Test loss: {test_loss:.4f}, Epochs w/o improvement: {not_improved}\n'
            full_message = f'Epoch: {epoch + 1}/{epochs_}, Loss: {epoch_loss}, ' + \
                           f'Test loss: {test_loss}, Epochs w/o improvement: {not_improved}\n'
        sys.stdout.flush()
        sys.stdout.write(message)

        with open(f'{path_to_save}/train_{model_name}.log', 'a') as f:
            f.write(full_message)
            f.close()

        # Early stopping
        if not_improved == patience:
            break

        epoch += 1

    torch.save(best_model, f'{path_to_save}/best-{model_name}.pt')
    return losses_, test_losses_
