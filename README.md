# automatic-cybersickness-detection
Automatic Cybersickness Detection by Deep Learning of Augmented Physiological Data from Off-the-Shelf Consumer-Grade Sensors

![alt text](https://github.com/m1237/automatic-cybersickness-detection/blob/main/cs_teaser.jpg?raw=true)

### Eye Tracking Data

The eye tracking data have been recorded with Unity and [SRanipal](https://forum.vive.com/topic/5642-sranipal-getting-started-steps/). The csv files contain the raw data recording from each session.

### Physiological Data

The physiological data was recorded by using two wearable sensory devices:
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
