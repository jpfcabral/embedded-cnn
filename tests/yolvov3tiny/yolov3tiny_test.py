import os
from datetime import datetime
from pathlib import Path
from tests.utils import init_test
from src.models.yolov3tiny.yolov3tiny import YoloV3Tiny

def test_yolvov3tiny_predict():
    init_test()

    yolov3tiny = YoloV3Tiny(weights_path="src/models/yolov3tiny/yolov3-tiny.weights",
                            config_path="src/models/yolov3tiny/yolov3-tiny.cfg",
                            image_size=(416, 416)
                            )
    
    print('\ndatetime,image,read_time,predict_time,objects')
    for image in os.listdir("./tests/images/person/images"):
        image_path = os.path.join("./tests/images/person/images/", image)
        objects = yolov3tiny.predict(image_path)
        print(f'{datetime.now()},{image},{round(yolov3tiny.read_image_time, 3)},{round(yolov3tiny.prediction_time, 3)},{len(objects)}')
