from siamese import SiameseNetwork
from siamese.predictors import LinearPredictor
from detector import FaceDetector
from data import ImageSelector, process_image
import numpy as np
import cv2
from utils.constants import RESNET_IMG_HEIGHT, RESNET_IMG_WIDTH


def web_cam(model_to_test):
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    selector = ImageSelector()
    names = []

    while True:
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_COMPLEX
        faces, start_xs, start_ys, end_xs, end_ys = detector.get_all_faces(frame)

        for face, start_x, start_y, end_x, end_y in zip(faces, start_xs, start_ys, end_xs, end_ys):
            try:
                face = cv2.resize(face, (RESNET_IMG_WIDTH, RESNET_IMG_HEIGHT))
                face = process_image(face, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    best_model_path = './siamese/models/triplet/resnet101.pt'
    predictor_path = './siamese/models/predictor-linear.pt'

    predictor = LinearPredictor().cuda().half()
    predictor.load(predictor_path)
    model = SiameseNetwork(base='resnet101') \
        .load(best_model_path) \
        .cuda() \
        .half() \
        .initialize(predictor, cuda=True, half=True)

    web_cam(model)
