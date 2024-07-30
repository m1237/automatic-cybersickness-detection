# automatic-cybersickness-detection
Automatic Cybersickness Detection by Deep Learning of Augmented Physiological Data from Off-the-Shelf Consumer-Grade Sensors

![alt text](https://github.com/m1237/automatic-cybersickness-detection/blob/main/cs_teaser.jpg?raw=true)

In this work, we used a VR environment that includes a rollercoaster to elicit cybersickness and used a
simple setup with sensory devices to get physiological responses. We deployed three different deep learning
models and one classical machine learning model to detect CS. Also, we realized a completely real-time
system using our best model. We demostrated that 4-layered bidirectional LSTM with data augmentation
gives superior results and this combination is the best solution for
sensor-based CS detection in real time applications particularly for wearable devices.



### Physiological Data

The physiological data was recorded by using three wearable sensory devices:

- Pico Neo 2 VR Headset (Tobii Ocumen):
    - `Eye Tracking (ET)`:
        - `Pupil Diameter`: 90 Hz, (left,right eye) values
        - `Gaze direction`: 90 Hz, (left,right eye; x,y,z) values
- Empatica E4 wristband: 
    - `Acceleration (ACC)`: 32 Hz, (x,y,z) values
    - `Electrodermal Activity (EDA)`: 4 Hz, skin conductance values in micro Siemens unit
    - `Photoplethysmography (PPG)`: 64 Hz, Blood Volume Pulse values  
    - `Heart Rate (HR)`: 1 Hz, beats per minute values
    - `Inter-beat interval (IBI)`: 64 Hz, time interval values between individual beats of the heart 
    - `Peripheral Body Temperature (TEMP)`: 4 Hz, temperature values in Celcius unit
- Polar H10 chest strap: 
    - `Acceleration (ACC)`: 200 Hz,  (x,y,z) values
    - `Electrocardiogram (ECG)`: 130 Hz, data values in millivolt unit


## Train 

Run 'train_model.py' for the training. For testing run 'evaluate_model.py'. Cross-validation should be done within train set.
Run 'cgan_modal/model.py' for data augmentation.


### Disclaimer: 

The code and materials provided in this repository are offered "as is" and without any warranties, either express or implied. The authors and contributors make no guarantees regarding the accuracy, reliability, or completeness of the code, documentation, or any related materials. The authors and contributors take no responsibility for the results produced by using this code. Any outcomes, findings, or conclusions drawn from the use of the code are solely the responsibility of the user. Users are advised to conduct their own testing and validation of the code to ensure it meets their requirements.

#### Research and Study Use
The code in this repository is intended for research and educational purposes only. It is not intended for use in production environments or as part of any commercial product or service. Users must obtain appropriate permissions and comply with any applicable laws and regulations when using the code for research or study.

#### Third-Party Content
This repository may include code, libraries, or other materials from third-party sources. The authors and contributors are not responsible for the content or functionality of any third-party materials. Users are advised to review the licenses and terms of use for any third-party content included in this repository.
