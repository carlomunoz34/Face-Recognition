from siamese import SiameseNetwork
from siamese.predictors import LinearPredictor
import torch.nn as nn
from data import FacesDataset, FaceTriplet
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datetime import datetime


def test_with_val_data(model_to_test: SiameseNetwork, predictor: LinearPredictor, cuda: bool = True,
                       half: bool = False) -> (np.ndarray, str):
    val_dataset = FacesDataset(False, True, base=model_to_test.base)
    batch_size = 64
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=14
    )
    set_length = len(val_dataset)
    all_predictions = np.zeros((set_length,))
    all_labels = np.zeros((set_length,))

    for idx, (first_face, second_face, labels) in tqdm(enumerate(val_loader), total=len(val_loader),
                                                       desc='Validation'):
        if cuda:
            first_face = first_face.cuda()
            second_face = second_face.cuda()

        if half:
            first_face = first_face.half()
            second_face = second_face.half()

        labels = labels.numpy()
        labels_len = len(labels)
        all_labels[idx * labels_len: (idx + 1) * labels_len] = labels

        norms = model_to_test(first_face, second_face)
        predictions = predictor.predict(norms.unsqueeze(1)).detach().cpu().numpy()
        predictions_len = len(predictions)
        all_predictions[idx * predictions_len: (idx + 1) * predictions_len] = np.round(predictions.squeeze())

    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)

    return cm, report


def test_loss(model_to_test: SiameseNetwork, cuda: bool = True):
    val_dataset = FaceTriplet(train=False, base='resnet')
    batch_size = 16
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=14
    )

    loss = nn.TripletMarginLoss()
    total_loss = 0
    for idx, (anchor, positive, negative) in tqdm(enumerate(val_loader), total=len(val_loader),
                                                  desc='Validation'):
        if cuda:
            anchor = anchor.cuda()
            positive = positive.cuda()
            negative = negative.cuda()

        anchor_vector = model_to_test.model(anchor)
        positive_vector = model_to_test.model(positive)
        negative_vector = model_to_test.model(negative)

        total_loss += loss(anchor_vector, positive_vector, negative_vector).item()
    return total_loss / len(val_dataset)


if __name__ == '__main__':
    model = SiameseNetwork(base='resnet101').cuda()
    # best_model_path = f'./models/triplet/resnet152_4096_ft_1_16.pt'
    best_model_path = f'./models/resnet101 4096.pt'
    model.load(best_model_path)
    model.eval()
    predictor_ = LinearPredictor().cuda()
    predictor_.load('./models/best-predictor-resnet152 2.pt')
    predictor_.eval()

    t0 = datetime.now()
    # conf_matrix, report_ = test_with_val_data(model, predictor_)
    print('Total loss:', test_loss(model))
    total_time = datetime.now() - t0
    print('Elapsed time:', total_time)
    # print(conf_matrix)
    # print(report_)
