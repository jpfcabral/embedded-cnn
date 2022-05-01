import cv2
import numpy as np
from src.predictor.predictor import Predictor


class YoloV5s(Predictor):
    def __init__(self, weights_path, image_size):
        super().__init__(image_size)

        self.__weights_path = weights_path
        self.model = self.__load_model(self.__weights_path)

        self.class_names = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
    
    def __load_model(self, weights_path):
        try:
            model = cv2.dnn.readNet(weights_path)
            return model
        except Exception as e:
            raise Exception(e)
    
    def __find_objects(self, layer_list: tuple, confidence_threshold=0.5):

        rows = layer_list.shape[0]

        image_width, image_height = self.image_size

        x_factor = image_width / 640
        y_factor =  image_height / 640

        results = []

        for r in range(rows):
            row = layer_list[r]
            confidence = row[4]
            if confidence >= confidence_threshold:

                classes_scores = row[5:]
                _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
                class_id = max_indx[1]
                if (classes_scores[class_id] > .25):
                    x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
                    left = int((x - 0.5 * w) * x_factor)
                    top = int((y - 0.5 * h) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    results.append([class_id, confidence, left, top, width, height])

        return results

    def predict(self, image_path):
        outputs = Predictor._predict(self, image_path=image_path)
        return self.__find_objects(layer_list=outputs[0])
