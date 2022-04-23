import os
from pathlib import Path
from tests.utils import init_test
from src.models.yolov3tiny.yolov3tiny import YoloV3Tiny

def test_yolvov3tiny_predict():
    init_test()

    yolov3tiny = YoloV3Tiny(weights_path=r"C:\Users\jpfca\Documents\projetos\embedded-cnn\src\models\yolov3tiny\yolov3-tiny.weights",
                            config_path=r"C:\Users\jpfca\Documents\projetos\embedded-cnn\src\models\yolov3tiny\yolov3-tiny.cfg",
                            image_size=(416, 416)
                            )
    
    for image in os.listdir("./tests/images/person/images"):
        image_path = os.path.join("./tests/images/person/images/", image)
        objects = yolov3tiny.predict(image_path)
        print(objects)
