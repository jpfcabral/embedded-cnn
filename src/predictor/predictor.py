import os
import time
import cv2

class Predictor:
    def __init__(self, image_size: tuple):

        self.image_size = image_size
        
        self.start_time = None
        self.read_image_time = None
        self.prediction_time = None

    def __read_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except:
            raise Exception("Error reading image")

    def _predict(self, image_path, layers = None):
        try:
            self.start_time = time.time()
            self.image = self.__read_image(image_path)
            self.read_image_time = time.time() - self.start_time
            
            blob = cv2.dnn.blobFromImage(self.image, 1 / 255, self.image_size, crop=False)
            self.model.setInput(blob)
            output = self.model.forward(layers)
            self.prediction_time = time.time() - (self.read_image_time + self.start_time)
            return output
        except:
            raise