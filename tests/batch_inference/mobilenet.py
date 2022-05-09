import os
from time import time
import cv2

PATH = 'tests/images/person/images'

mobilenet = cv2.dnn.readNetFromCaffe(
    'src/models/mobilenet/MobileNetSSD_deploy.prototxt',
    'src/models/mobilenet/MobileNetSSD_deploy.caffemodel'
)

def find_objects(layer_list: tuple, confidence_threshold=0.5):
    #Size of frame resize (300x300)
    cols = 300
    rows = 300

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

def test_batch_size_2():
    print(f'number of threads detected: {cv2.getNumThreads()}')

    for BATCH_SIZE in range(1,6):
        images = []
        time_elapesed_count = []
        count = 0

        for image_path in os.listdir(PATH):
            img = cv2.imread(os.path.join(PATH, image_path))
            images.append(img)
            if len(images) == BATCH_SIZE:
                count += 1
                start_time = time()
                blobs = cv2.dnn.blobFromImages(images, 1 / 255, (300,300), crop=False)
                inputs = mobilenet.setInput(blobs)
                outputs = mobilenet.forward()
                end_time = time() - start_time
                time_elapesed_count.append(end_time)
                images = []

        av_time = sum(time_elapesed_count)/count

        print(
            f'batch size: {BATCH_SIZE}',
            f' time/batch: {round(av_time, 4)}',
            f' time/image: {round(av_time/BATCH_SIZE, 4)}'
        )

test_batch_size_2()