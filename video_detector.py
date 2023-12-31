import cv2
import numpy as np
import tensorflow as tf

"""
    What is Interpreter in Tflite ?
    In iTensorFlow Lite (TFLite), an Interpreter s a component used to execute and run TensorFlow Lite models. 
    It's essentially an interface that facilitates the execution of the model, handling input data, producing output, and managing the 
        overall process of running inference on a TFLite model.
"""


# tflite model
interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# input and output 
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

"""
    !!! class names they must be in the same order with your labelmap.txt file (This file is created when model is training) 
"""
labels = ['bir', 'bes', 'on', 'yirmibes', 'elli']  

 # viode file that you want to see
video_path = r"C:\Users\sirom\Downloads\VID_20231231_143333.mp4" # !! here you need to write your videos path
# Open video file 
cap = cv2.VideoCapture(video_path)

# size of screen , 320x320 because or model expect input in this dimensions
new_width, new_height = 320, 320  

while cap.isOpened(): # Returns true if video capturing has been initialized already.
    
    """
        read() : Grabs, decodes and returns the next video frame. It return 2 value  ,  it returns boolean  and data
        Boolean value indicates whether the read operation was successful or not. (first one)
    """
    ret, frame = cap.read()  

    if not ret:
        break
    
    # resizing video
    frame = cv2.resize(frame, (new_height, new_width))

    # change format of frame
    input_data = cv2.resize(frame, (new_width, new_height))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = (np.float32(input_data) - 127.5) / 127.5  # normalize data

    # input for tflite model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # detection part
    boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # coordinates
    classes = interpreter.get_tensor(output_details[3]['index'])[0]  # class indexes
    scores = interpreter.get_tensor(output_details[0]['index'])[0]  # scores    
    detections = [] 

    
    # draw rects and labels
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            ymin = int(max(1, (boxes[i][0] * new_height)))
            xmin = int(max(1, (boxes[i][1] * new_width)))
            ymax = int(min(new_height, (boxes[i][2] * new_height)))
            xmax = int(min(new_width, (boxes[i][3] * new_width)))



            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10),
                           (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

    print(detections)

    
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik yapın ve pencereleri kapatın
cap.release()
cv2.destroyAllWindows()

