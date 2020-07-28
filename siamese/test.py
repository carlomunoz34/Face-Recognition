from siamese import FaceRecognition
from detector import FaceDetector
from data import ImageSelector, process_image, FacesDataset
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import cv2


def test_with_val_data(model_to_test: FaceRecognition, cuda: bool = True) -> (np.ndarray, str):
    val_dataset = FacesDataset(False, True)
    batch_size = 128
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size
    )
    set_length = len(val_dataset)
    all_predictions = np.zeros((set_length,))
    all_labels = np.zeros((set_length,))

    for idx, (first_face, second_face, labels) in tqdm(enumerate(val_loader), total=len(val_loader),
                                                       desc='Validation'):
        if cuda:
            first_face = first_face.cuda()
            second_face = second_face.cuda()

        labels = labels.numpy()
        labels_len = len(labels)
        all_labels[idx * labels_len: (idx + 1) * labels_len] = labels

        predictions = model_to_test.predict(first_face, second_face).cpu().numpy()
        predictions_len = len(predictions)
        all_predictions[idx * predictions_len: (idx + 1) * predictions_len] = predictions

    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions)

    return cm, report


def test_with_web_cam(model_to_test):
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    selector = ImageSelector()

    while True:
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_PLAIN
        faces, start_xs, start_ys, end_xs, end_ys = detector.get_all_faces(frame)

        for face, start_x, start_y, end_x, end_y in zip(faces, start_xs, start_ys, end_xs, end_ys):
            try:
                face = process_image(face, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            except:
                break
            face = np.expand_dims(face, 0)
            prediction = model_to_test.get_database_prediction(face)
            if prediction != -1:
                name = selector.get_name_by_index(prediction)
            else:
                name = 'Unknown'

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(frame, name, (start_x, start_y), font, 2, (255, 0, 0), 2)

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    best_model_path = 'models/mobilenet_3x512_300.pt'
    model = FaceRecognition() \
        .load(best_model_path) \
        .cuda() \
        .initialize(cuda=True) \
        .eval()

    # conf_matrix, report = test_with_val_data(model)
    # print(conf_matrix)
    # print(report)

    test_with_web_cam(model)
