import os
from datetime import datetime
from pathlib import Path
from tests.utils import init_test
from src.models.mobilenet.mobilenet import MobileNet

try:
    from gpiozero import CPUTemperature
    cpu = CPUTemperature()
    temp_flag = True
except:
    cpu = None

def test_mobilenet_predict():
    init_test()

    mobilenet = MobileNet(weights_path='src/models/mobilenet/MobileNetSSD_deploy.caffemodel', 
                            config_path='src/models/mobilenet/MobileNetSSD_deploy.prototxt', 
                            image_size=(300,300))
    
    print('\ndatetime,image,read_time,predict_time,objects,temperature')
    for image in os.listdir("./tests/images/person/images"):
        image_path = os.path.join("./tests/images/person/images/", image)
        objects = mobilenet.predict(image_path)

        temperature = cpu.temperature if cpu is not None else 0.0
        print(f'{datetime.now()},{image},{round(mobilenet.read_image_time, 3)},{round(mobilenet.prediction_time, 3)},{len(objects)},{temperature}')
