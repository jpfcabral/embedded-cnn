import cv2
import numpy as np
from src.predictor.predictor import Predictor


class YoloV3Tiny(Predictor):
    def __init__(self, weights_path, config_path, image_size):
        super().__init__(image_size)

        self.__weights_path = weights_path
        self.__config_path = config_path
        self.model = self.__load_model(self.__config_path, self.__weights_path)

        self.model_layer_names = self.model.getLayerNames()
        self.model_disconnected_layers = [self.model_layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]

    
    def __load_model(self, config_path, weights_path):
        try:
            model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
            return model
        except Exception as e:
            raise Exception(e)
    
    def __find_objects(self, layer_list: tuple, confidence_threshold=0.5):
        detected_objects = []
        for layer in layer_list:
            for bounding_boxes_values in layer:

                # Guarda os valores de confidence para cada classe armazenados depois do quinto valor da lista
                detection_scores = bounding_boxes_values[5:]

                # Retorna a posição do maior valor numa lista de numeros (conficence)
                class_id = np.argmax(detection_scores)

                # Armazena o valor numerico do confidence
                confidence = round(detection_scores[class_id], 2)

                # Verifica se a classe com maior confiança é maior que nosso limite de confiança
                if confidence > confidence_threshold:
                    x = bounding_boxes_values[0] if bounding_boxes_values[0] > 0 else 0
                    y = bounding_boxes_values[1] if bounding_boxes_values[1] > 0 else 0
                    w = bounding_boxes_values[2]
                    h = bounding_boxes_values[3]
                    detected_objects.append([class_id, confidence, x, y, w, h])

        return detected_objects

    def predict(self, image_path):
        outputs = Predictor._predict(self, image_path=image_path, layers=self.model_disconnected_layers)
        return self.__find_objects(layer_list=outputs)