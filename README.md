
<h1>  Yolov7 Model</h1>
I trained other model with same dataset , this time I used Yolov7 model , results were more accurate , you can check it : https://github.com/siromermer/Yolov7-CustomModel-Money-Counter-TurkishCurrency/blob/master/README.md

<br><br>
<h1>  SSD-MobileNet-v2-FPNLite-320 Model</h1>
<br>
<H2> Folders</H2>
images_for_testing : Images that you want to test <br>
test_results : result of your test images (labeled ,annotated images)<br>
detect.tflite : tflite model<br>
labelmap.txt : labels <br>
photo_detector.py : for testing new images execute this file<br>
video_detector.py : for testing model in video use this file <br><br>

<H2> Model & Environment </H2>
I trained a model that counts Turkish currency, it's a money counter.<br>
In near future , i will use this model in some Mobile App that is why i trained this model<br>

I followed this notebook for setting up TensorFlow 2 Object Detection API environment --> https://colab.research.google.com/github/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb<br>

My Google Collab Notebook -->  https://colab.research.google.com/drive/1CtEMYm5szfNkrkj4Tg2-S2_opbPE3lbZ#scrollTo=OzlL4-ROf3gI 



# Collection of Data
I collected 208 images of money using my phone camera. These images showcase various backgrounds and different combinations of currency notes and coins (for instance, 3 units of 1 Turkish Lira, 5 units of 0.25 Turkish Lira...).<br> That was my first Object Detection Model therefore i didnt spend so many time for taking photo but i will add more images to my data and train it again in future<br>
For image Annotation i used  LabelImg --> https://github.com/HumanSignal/labelImg<br>
I draw more than 1000  rect box , it may be the most time consuming part for me<br>

Example Images from my dataset(not anotated image)<br>
 ![ornek](https://github.com/siromermer/Money-Counter-TurkishCurrency/assets/113242649/68b711a5-5e8a-4e81-978a-2b720c51de46)
<br><br><br>
Example Anotated Images (i take this ss when drawing boxes in LabelImg)
<br>
![ornek3](https://github.com/siromermer/Money-Counter-TurkishCurrency/assets/113242649/5807265f-2eb4-4dc0-95dc-645dd6391b3d)
<br><br><br>
 
I used TensorFlow 2 Object Detection API to train an SSD-MobileNet <br>

# SSD-MobileNet-v2-FPNLite-320 <br>
SSD-MobileNet-v2-FPNLite-320 is a specific neural network architecture designed for object detection tasks.<br>
SSD (Single Shot Multibox Detector): It's an object detection algorithm that performs both object localization and classification in a single forward pass of the network.<br>
MobileNetV2: This serves as the base network architecture. <br>
FPNLite : FPN is a technique used to improve object detection at various scales. <br>

This architecture is tailored for real-time or edge device applications where computational resources are limited, aiming to strike a balance between accuracy and speed .<br>

# Training Model
I trained model in Google Collab , therefore there was some restrictions ,  for example one session must be take 12 hours at max  , and training and other operations(prep data , prep environment takes more than 12 hours) takes so many time <br>
Because of time restriction i didnt train model as needed , number of steps  was 20000 but in the end model was still learning as you can see from below graphes   <br>
when i create environment in my own PC i will train model with more steps and accuracy will increase for sure
![Ekran görüntüsü 2023-12-30 205954](https://github.com/siromermer/Money-Counter-TurkishCurrency/assets/113242649/0e130ccb-2ed0-48e0-8dd9-859add1c5d95)
<br><br>  <br>Screen shot during the model training <br>
![Ekran görüntüsü 2023-12-30 210210](https://github.com/siromermer/Money-Counter-TurkishCurrency/assets/113242649/92f31066-af02-456a-9c5a-283be1adf5e8)
 
<br>

# Example Images after Training 
I test Model with compeletly new images that i take after training model again with my phone<br>
Model seems good for now because data is not that much and i trained less than i have to train(because of restriction of Google Collab)
<br><br>
Observations : The model can sometimes confuse 10 kurus coin with 25 kurus coins , i will give attention to this 2 coin when i gather data again  
<br>
<br>
![trmoney12](https://github.com/siromermer/Money-Counter-TurkishCurrency/assets/113242649/869df338-109a-43f3-ab6f-4a09b57af2d9)

![trmoney3](https://github.com/siromermer/Money-Counter-TurkishCurrency/assets/113242649/a0be252f-1c8b-4272-9087-b70bbb2ce02b)

![trmoney4](https://github.com/siromermer/Money-Counter-TurkishCurrency/assets/113242649/be7102f2-281e-4b90-8dc6-d40637743fb4)
