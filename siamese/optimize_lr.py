from torch.utils.data import DataLoader
from siamese import SiameseNetwork
from data import FaceTriplet
from siamese.train import train
from siamese.train_triplet_loss import get_triplet_step
import numpy as np


def random_search(iterations, low, high):
    batch_size = 32
    epochs = 10
    start_epoch = 0
    triplet_step_ = get_triplet_step(True)
    path = './models/triplet'
    base = 'resnet101'

    train_data = FaceTriplet(train=True, base=base)
    train_loader = DataLoader(train_data, batch_size, False)

    test = FaceTriplet(train=False, base=base)
    test_loader = DataLoader(test, batch_size, False)

    best_loss = float('inf')

    for i in range(iterations):
        siamese_network = SiameseNetwork(base=base).cuda()
        lr_log = np.random.uniform(low, high)
        lr = 10 ** -lr_log
        model_name_ = f'{siamese_network.name}_{lr_log}'

        print(f'Iteration: {i}, training with: {lr}')
        with open(f'lr.log', 'a') as f:
            f.write(f'Iteration: {i}, training with: {lr_log}\n')
            f.close()

        losses, test_losses = train(siamese_network, train_loader, test_loader, epochs, triplet_step_, lr,
                                    model_name_, start_epoch_=start_epoch, adam=False, patience=epochs,
                                    path_to_save=path)

        current_best_loss = min(test_losses).item()

        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_lr_log = lr_log

            with open(f'lr.log', 'a') as f:
                f.write(f'Best log:\n{best_lr_log}\n')
                f.close()

            np.array([best_lr_log], dtype=np.float32).tofile('best_lr.bin')


if __name__ == '__main__':
    random_search(5, 1.2, 2.2)
