from torch.nn import TripletMarginLoss
from torch.utils.data import DataLoader
from siamese import SiameseNetwork
from data import FaceTriplet
import sys
import matplotlib.pyplot as plt
from siamese.train import train
import numpy as np


def get_triplet_step(cuda=True):
    loss = TripletMarginLoss()

    def triplet_step(model, anchor, positive, negative):
        if cuda:
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

        anchor_vector = model.model(anchor)
        positive_vector = model.model(positive)
        negative_vector = model.model(negative)

        return loss(anchor_vector, positive_vector, negative_vector)
    return triplet_step


def triplet_train():
    # lr = 10 ** -np.fromfile("./best_lr.bin", dtype=np.float32)[0]
    lr = 0.001
    ft_lr = lr * 0.01
    batch_size = 8
    epochs = -1
    start_epoch = 0
    triplet_step_ = get_triplet_step(True)
    path = './models/triplet'
    num_workers = 14

    bases = ['wide_resnet']
    for base in bases:
        train_data = FaceTriplet(train=True, base='resnet')
        train_loader = DataLoader(train_data, batch_size, False, num_workers=num_workers)

        test = FaceTriplet(train=False, base='resnet')
        test_loader = DataLoader(test, batch_size, False, num_workers=num_workers)

        siamese_network = SiameseNetwork(base=base).cuda()
        model_name_ = siamese_network.name

        sys.stdout.write(f"Starting train of {base}\n")
        losses, test_losses = train(siamese_network, train_loader, test_loader, epochs, triplet_step_, lr,
                                    model_name_, start_epoch_=start_epoch, adam=True, patience=10, path_to_save=path)

        plt.plot(losses, label='Train losses')
        plt.plot(test_losses, label='Test losses')
        plt.legend()
        plt.savefig(f"losses_{siamese_network.name}.png")
        plt.clf()

        # Fine tune the model
        for i in range(1, 5):
            sys.stdout.write(f"Starting fine tune {i} of {base}\n")
            ft_epochs = -1
            ft_start_epoch = 0
            if i == 1:
                siamese_network.load(f'./models/triplet/best-{model_name_}.pt')
            else:
                siamese_network.load(f'./models/triplet/best-{model_name_}_ft_{i - 1}.pt')
            siamese_network.deep_fine_tune(i)
            model_name_ft = model_name_ + '_ft_' + str(i)

            losses, test_losses = train(siamese_network, train_loader, test_loader, ft_epochs, triplet_step_, ft_lr,
                                        model_name_ft, start_epoch_=ft_start_epoch, adam=True, patience=10,
                                        path_to_save=path)

            plt.plot(losses, label='Train losses')
            plt.plot(test_losses, label='Test losses')
            plt.legend()
            plt.savefig(f"losses_{model_name_ft}.png")
            plt.clf()


if __name__ == '__main__':
    triplet_train()
