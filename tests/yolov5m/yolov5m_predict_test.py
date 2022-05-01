import os
from datetime import datetime
from pathlib import Path
from tests.utils import init_test
from src.models.yolov5m.yolov5m import YoloV5m

try:
    from gpiozero import CPUTemperature
    cpu = CPUTemperature()
    temp_flag = True
except:
    cpu = None

def test_yolov5m_predict():
    init_test()

    yolov5m = YoloV5m(weights_path='src/models/yolov5m/yolov5m.onnx', 
                        image_size=(640,640))
    
    print('\ndatetime,image,read_time,predict_time,objects,temperature')
    for image in os.listdir("./tests/images/person/images"):
        image_path = os.path.join("./tests/images/person/images/", image)
        objects = yolov5m.predict(image_path)

        temperature = cpu.temperature if cpu is not None else 0.0
        print(f'{datetime.now()},{image},{round(yolov5m.read_image_time, 3)},{round(yolov5m.prediction_time, 3)},{len(objects)},{temperature}')
