# automatic-cybersickness-detection
Automatic Cybersickness Detection by Deep Learning of Augmented Physiological Data from Off-the-Shelf Consumer-Grade Sensors

![alt text](https://github.com/m1237/automatic-cybersickness-detection/blob/main/cs_teaser.jpg?raw=true)

In this work, we used a VR environment that includes a rollercoaster to elicit cybersickness and used a
simple setup with sensory devices to get physiological responses. We deployed three different deep learning
models and one classical machine learning model to detect CS. Also, we realized a completely real-time
system using our best model. We demostrated that 4-layered bidirectional LSTM with data augmentation
gives superior results and this combination is the best solution for
sensor-based CS detection in real time applications particularly for wearable devices.

### Eye Tracking Data

The eye tracking data have been recorded with Unity and [SRanipal](https://forum.vive.com/topic/5642-sranipal-getting-started-steps/). The csv files contain the raw data recording from each session.

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
