import cv2
import numpy as np
import tensorflow as tf
import glob
from tensorflow.lite.python.interpreter import Interpreter
import random
import matplotlib.pyplot as plt

testimage_path='images_for_testing/'   # Path to test images folder
model_path='detect.tflite'   # Path to .tflite model file
labels_path='labelmap.txt'  # change this to your path


images = glob.glob(testimage_path + '/*.jpg') + glob.glob(testimage_path + '/*.jpeg') + glob.glob(testimage_path + '/*.png') + glob.glob(testimage_path + '/*.bmp')

with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Load the Tensorflow Lite model into memory
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

 # Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Randomly select test images
num_test_images=3
images_to_test = random.sample(images, num_test_images) # Chooses k unique random elements from a population sequence or set.


# Loop over every image and perform detection
for image_path in images_to_test:

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    detections = []

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])    
    
    a=random.randint(1,10000)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12,16))
    plt.imshow(image)
    plt.savefig(f'test_results/trmoney{a}.jpeg',bbox_inches='tight')
    plt.show()
    


