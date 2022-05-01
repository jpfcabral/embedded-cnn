import cv2
import numpy as np
from src.predictor.predictor import Predictor


class MobileNet(Predictor):
    def __init__(self, weights_path, config_path, image_size):
        super().__init__(image_size)

        self.__weights_path = weights_path
        self.__config_path = config_path
        self.model = self.__load_model(self.__config_path, self.__weights_path)

        self.class_names = { 0: 'background',
            1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
            5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
            10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
            14: 'motorbike', 15: 'person', 16: 'pottedplant',
            17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }
    
    def __load_model(self, config_path, weights_path):
        try:
            model = cv2.dnn.readNetFromCaffe(config_path, weights_path)
            return model
        except Exception as e:
            raise Exception(e)
    
    def __find_objects(self, layer_list: tuple, confidence_threshold=0.5):
            #Size of frame resize (300x300)
        cols = self.image_size[1] 
        rows = self.image_size[0] 

        #For get the class and location of object detected, 
        # There is a fix index for class, location and confidence
        # value in @layer_list array .
        results = []
        for i in range(layer_list.shape[2]):
            confidence = layer_list[0, 0, i, 2] #Confidence of prediction 
            if confidence > confidence_threshold: # Filter prediction 
                class_id = int(layer_list[0, 0, i, 1]) # Class label

                # Object location 
                xLeftBottom = int(layer_list[0, 0, i, 3] * cols) 
                yLeftBottom = int(layer_list[0, 0, i, 4] * rows)
                xRightTop   = int(layer_list[0, 0, i, 5] * cols)
                yRightTop   = int(layer_list[0, 0, i, 6] * rows)

                results.append([class_id, confidence, xLeftBottom, yLeftBottom, xRightTop, yRightTop])
        return results

    def predict(self, image_path):
        outputs = Predictor._predict(self, image_path=image_path)
        return self.__find_objects(layer_list=outputs)
