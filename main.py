from src.models import FaceDetector, FaceLandmarkDetector, GazeEstimator, HeadPoseEstimator
from src.mouse_controller import MouseController
from src.input_feeder import InputFeeder

from argparse import ArgumentParser
import cv2
from math import sqrt

import time
import logging as log
import sys

def build_parser():
    '''
    Description:
        Builds the Argument Parser which takes command line inputs from user.
    Params:
        None
    Returns:
        parser: Argument Parser Object with argument variables added.
    '''

    parser = ArgumentParser()

    input_type_desc = "Give the type of input stream"
    input_stream_desc = "Give path of input stream if input_type is 'video' or 'image'"
    device_desc = "State the device on which inference should happen"
    conf_desc = "Probability threshold for face detections filtering"

    parser.add_argument('-input_stream', help = input_stream_desc, type = str, default = None)
    parser.add_argument('-device', choices=['CPU','GPU', 'HETERO:FPGA,CPU', 'HETERO:MYRIAD,CPU'], help = device_desc, default='CPU', type = str)
    parser.add_argument('-prob_threshold', type=float, default=0.5, help = conf_desc)

    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-input_type', help = input_type_desc, choices=['cam','video','image'], required=True)

    return parser

def main(args):
    '''
    Description:
        Captures frames from the input stream i.e CAM, VIDEO or IMAGE and runs inference
    '''

    #Initialize the Mouse Controller object
    controller = MouseController('high','fast')

    #Initialize the Input Feeder object
    feed = InputFeeder(input_type = args.input_type, input_file = args.input_stream)
    feed.load_data()

    #Loading the models

    start_time = time.time()

    faceDetector = FaceDetector(model_name = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001",conf=args.prob_threshold, device = args.device)
    faceDetector.load_model()
    prev_time = time.time()
    log.info("Face Detection Model Loading time:" + str(prev_time-start_time))

    faceLandmarkDetector = FaceLandmarkDetector(model_name = "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009", device = args.device)
    faceLandmarkDetector.load_model()
    modelLoadingTime = time.time() - prev_time
    log.info("Face Landmark Detection Model Loading time:" + str(modelLoadingTime))
    prev_time = modelLoadingTime + prev_time

    headPoseEstimator = HeadPoseEstimator(model_name = "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001", device = args.device)
    headPoseEstimator.load_model()
    modelLoadingTime = time.time() - prev_time
    log.info("HeadPose Detection Model Loading time:" + str(modelLoadingTime))
    prev_time = modelLoadingTime + prev_time

    gazeEstimator = GazeEstimator(model_name = "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002", device = args.device)
    gazeEstimator.load_model()
    modelLoadingTime = time.time() - prev_time
    log.info("Gaze Estimation Model Loading time:" + str(modelLoadingTime))
    prev_time = modelLoadingTime + prev_time

    log.info("Total Model Loading Time" + str(time.time() - start_time))

    for batch in feed.next_batch():

        key_pressed = cv2.waitKey(60)
 
        try: 
            face, coords = faceDetector.predict(batch)
            eye_width = sqrt((face.shape[0]*face.shape[1])/100)

            batch = cv2.rectangle(batch, (coords[0],coords[1]), (coords[2],coords[3]), (0,255,0), thickness = 2)

            left_eye_X, left_eye_Y, right_eye_X, right_eye_Y = faceLandmarkDetector.predict(face)

            left_eye_X, left_eye_Y, right_eye_X, right_eye_Y, eye_width = int(left_eye_X), int(left_eye_Y), int(right_eye_X), int(right_eye_Y), int(eye_width)

            face = cv2.rectangle(face, (int(left_eye_X-eye_width), int(left_eye_Y-eye_width)), (int(left_eye_X+eye_width), int(left_eye_Y+eye_width)), color = (255,0,0), thickness = 1)
            face = cv2.rectangle(face, (int(right_eye_X-eye_width), int(right_eye_Y-eye_width)), (int(right_eye_X+eye_width), int(right_eye_Y+eye_width)), color = (255,0,0), thickness = 1)

            left_eye = face[left_eye_Y-eye_width:left_eye_Y+eye_width, left_eye_X-eye_width:left_eye_X+eye_width]
            right_eye = face[right_eye_Y-eye_width:right_eye_Y+eye_width, right_eye_X-eye_width:right_eye_X+eye_width]
            
            headpose_angles = headPoseEstimator.predict(face)

            gaze_vector = gazeEstimator.predict(left_eye, right_eye, headpose_angles)

            controller.move(gaze_vector[0][0], gaze_vector[0][1])
        
        except:
            cv2.putText(batch, "Landmarks or Face not Detected!", org = (10,50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, thickness = 2, fontScale = 1, color = (0,0,255))

        cv2.imshow("Output", batch)
        if key_pressed == 27:
           break

    feed.close()

if __name__ == "__main__": 

    args = build_parser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO, stream=sys.stdout)
    main(args)