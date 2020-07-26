from siamese.SiameseNetwork import FaceRecognition
from siamese.FacesDataset import FacesDataset
from glob import glob
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def test_with_val_data(model_to_test: FaceRecognition, cuda: bool = True) -> np.ndarray:
    val_dataset = FacesDataset(False, True)
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=128
    )
    cm = np.zeros((2, 2))

    for first_face, second_face, labels in tqdm(val_loader, total=len(val_loader),
                                                desc='Validation'):
        if cuda:
            first_face = first_face.cuda()
            second_face = second_face.cuda()

        labels = labels.numpy()

        predictions = model_to_test.predict(first_face, second_face).cpu().numpy()
        cm_ = confusion_matrix(labels, predictions)
        cm += cm_

    return cm


def test_with_web_cam():
    pass


if __name__ == '__main__':
    best_model_path = glob('./models/*')[-1]
    model = FaceRecognition().load(best_model_path).cuda()

    conf_matrix = test_with_val_data(model)
    print(conf_matrix)
