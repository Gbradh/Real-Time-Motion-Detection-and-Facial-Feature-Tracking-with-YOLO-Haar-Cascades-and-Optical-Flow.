# Real-Time Motion Detection and Facial Feature Tracking using YOLO, Haar Cascades and Optical Flow.

## Introduction
 This project presents an advanced system for real-time motion detection and facial 
feature tracking, utilizing a combination of YOLO, Haar cascades, and optical flow 
techniques. The integration of these methods provides robust and accurate detection 
and tracking capabilities for various applications, including surveillance and human
computer interaction. The system captures video input, detects motion through 
background subtraction and frame differencing, and tracks facial features using pre
trained Haar cascades. Additionally, it leverages YOLO for object detection and 
Lucas-Kanade optical flow for tracking movements across frames. The result is a 
sophisticated, real-time visual system capable of detailed analysis and responsive 
tracking.

yolo: https://docs.ultralytics.com/

yolov3: https://github.com/ultralytics/yolov3

Harcascade: https://github.com/opencv/opencv/tree/master/data/haarcascades

Harcascade: http://pyimagesearch.com/2021/04/12/opencv-haar-cascades/

Optical Flow: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html


## Procedure
To develop and test a real-time system for detecting motion and tracking facial 
features using a combination of YOLO, Haar cascades, and optical flow techniques.


### Materials:

 • Computer with Python installed
 
 • Webcam for video capture
 
 • Pre-trained YOLO model files (yolov3.weights, yolov3.cfg, and coco.names)
 
 • OpenCV library for image processing
 
 • Pandas for data manipulation
 
 • NumPy for numerical operations
 
 • Haar cascades XML files for face, eye, and spectacles detection Setup
 
 • Install Required Libraries:
 
 • Bash: pip install opencv-python pandas numpy
 
 • Prepare YOLO Model Files: Download the yolov3.weights, yolov3.cfg, and coco.names files from the YOLO website.

 • Load Haar Cascades: Use the pre-trained Haar cascades included with OpenCV for face, eye, and spectacles detection.



### Procedure 

 • Initialize the Environment: Import necessary libraries: cv2, pandas, datetime, numpy
 
 • Setup Data Structures: Initialize a DataFrame to store the start and end times of 
detected motion.
 
 • Load Detection Models:
 
 • Load Haar cascades for face, eye, and spectacles detection
 
 • Load YOLO model for object detection.
 
 • Capture Video: Initialize video capture and set frame rate
 
 • Background Subtraction and Optical Flow Setup: Initialize background subtractor and set parameters for Lucas-Kanade optical flow.
 
 • Process Video Frames: Continuously read frames from the video feed and process them for motion and facial feature detection.
 
 • Store Motion Data: Create a DataFrame to store the start and end times of detected motion.
 
 • Save Results: Save the motion data to a CSV file.
 
 • Clean Up: Release the video capture and close all OpenCV windows


 ![image](https://github.com/user-attachments/assets/7b8c7e55-546b-4ad8-aa45-f81fe6986df1)


## Conclusion   

This experiment demonstrates the integration of multiple computer vision 
techniques to achieve robust and accurate detection and tracking, suitable for 
applications such as surveillance and human-computer interaction. 
