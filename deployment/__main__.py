import torch.cuda

from siamese import SiameseNetwork
from detector import FaceDetector
from data import ImageSelector, process_image
import numpy as np
import cv2
from utils.constants import IMG_HEIGHT, IMG_WIDTH, mean, std


def web_cam(model_to_test):
    cap = cv2.VideoCapture(1)
    detector = FaceDetector()
    selector = ImageSelector()
    names = []

    while True:
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_COMPLEX
        faces, start_xs, start_ys, end_xs, end_ys = detector.get_all_faces(frame)

        for face, start_x, start_y, end_x, end_y in zip(faces, start_xs, start_ys, end_xs, end_ys):
            try:
                face = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))
                face = process_image(face, mean=mean, std=std)
            except:
                break
            face = np.expand_dims(face, 0).astype(np.float16)
            prediction, prob = model_to_test.get_database_prediction(face)
            if prob > 0.5:
                name = selector.get_name_by_index(prediction.item())
                if name not in names:
                    names.append(name)
                    print(name)
            else:
                name = 'Unknown'

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(frame, f"{name}:{prob * 100: .2f}%", (start_x, start_y - 2), font, 0.5, (255, 255, 255), 1)

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model_path = './siamese/models/best-b3_with_predictor.pt'

    model = SiameseNetwork(device, best_model_path) \
        .load(best_model_path) \
        .cuda() \
        .initialize()

    web_cam(model)
