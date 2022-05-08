import os
from datetime import datetime
from pathlib import Path
from tests.utils import init_test
from src.models.yolov5s.yolov5s import YoloV5s

try:
    from gpiozero import CPUTemperature
    cpu = CPUTemperature()
    temp_flag = True
except:
    cpu = None

def test_yolov5s_predict():
    init_test()

    yolov5s = YoloV5s(weights_path='src/models/yolov5s/yolov5s.onnx', 
                        image_size=(640,640))
    
    print('\ndatetime,image,read_time,predict_time,objects,temperature')
    for image in os.listdir("./tests/images/person/images"):
        image_path = os.path.join("./tests/images/person/images/", image)
        objects = yolov5s.predict(image_path)

        temperature = cpu.temperature if cpu is not None else 0.0
        print(f'{datetime.now()},{image},{round(yolov5s.read_image_time, 3)},{round(yolov5s.prediction_time, 3)},{len(objects)},{temperature}')
