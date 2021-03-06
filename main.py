from src.models import FaceDetector, FaceLandmarkDetector, GazeEstimator, HeadPoseEstimator
from src.mouse_controller import MouseController
from src.input_feeder import InputFeeder

from argparse import ArgumentParser, RawTextHelpFormatter
import cv2
from math import sqrt, sin, cos, pi

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

    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    input_type_desc = "Give the type of input stream"
    input_stream_desc = "Give path of input stream if input_type is 'video' or 'img'"
    device_desc = "State the device on which inference should happen"
    conf_desc = "Probability threshold for face detections filtering"
    flag_desc = "Choose a particular model only to run inference \n fd: Only Face Detection \n ld: Only Face Landmark Detection \n hd: Only Head Pose Detection \n ge: Only Gaze Estimation"

    parser.add_argument('-input_stream', help = input_stream_desc, type = str, default = None)
    parser.add_argument('-device', choices=['CPU','GPU', 'HETERO:FPGA,CPU', 'HETERO:MYRIAD,CPU'], help = device_desc, default='CPU', type = str)
    parser.add_argument('-prob_threshold', type=float, default=0.5, help = conf_desc)
    parser.add_argument('-flag', choices=['fd','ld','hd','ge'], default = None, help = flag_desc, type = str)

    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('-input_type', help = input_type_desc, choices=['cam','video','img'], required=True)
    

    return parser

def main(args):
    '''
    Description:
        Captures frames from the input stream i.e CAM, VIDEO or img and runs inference
    '''

    flag = args.flag

    # Initializing the flags
    fdFlag = False; ldFlag = False; hdFlag = False; geFlag = False; mcFlag = False

    # Setting up the flags
    if flag == None: mcFlag = True
    elif flag == 'fd': fdFlag = True
    elif flag == 'hd': hdFlag = True
    elif flag == 'ld': ldFlag = True
    elif flag == 'ge': geFlag = True

    # Initialize the Mouse Controller object
    controller = MouseController('high','fast')

    # Initialize the Input Feeder object
    feed = InputFeeder(input_type = args.input_type, input_file = args.input_stream)
    feed.load_data()

    # Loading the models

    start_time = time.time()

    faceDetector = FaceDetector(model_name = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001",conf=args.prob_threshold, device = args.device)
    faceDetector.load_model()
    prev_time = time.time()
    log.info("Face Detection Model Loading time: " + str(prev_time-start_time))

    if ldFlag or mcFlag or geFlag:
        faceLandmarkDetector = FaceLandmarkDetector(model_name = "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009", device = args.device)
        faceLandmarkDetector.load_model()
        modelLoadingTime = time.time() - prev_time
        log.info("Face Landmark Detection Model Loading time: " + str(modelLoadingTime))
        prev_time = modelLoadingTime + prev_time

    if hdFlag or mcFlag or geFlag:
        headPoseEstimator = HeadPoseEstimator(model_name = "models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001", device = args.device)
        headPoseEstimator.load_model()
        modelLoadingTime = time.time() - prev_time
        log.info("HeadPose Detection Model Loading time: " + str(modelLoadingTime))
        prev_time = modelLoadingTime + prev_time

    if geFlag or mcFlag:
        gazeEstimator = GazeEstimator(model_name = "models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002", device = args.device)
        gazeEstimator.load_model()
        modelLoadingTime = time.time() - prev_time
        log.info("Gaze Estimation Model Loading time: " + str(modelLoadingTime))
        prev_time = modelLoadingTime + prev_time

    log.info("Total Model Loading Time: " + str(time.time() - start_time))

    for batch in feed.next_batch():

        key_pressed = cv2.waitKey(60)
 
        try: 

            '''	  
            Face Box: Lower Point and Higher Point Co-ordinates	  
            coords = [xmin, ymin, xmax, ymax]
            '''

            face, coords = faceDetector.predict(batch)
            eye_width = sqrt((face.shape[0]*face.shape[1])/100)

            #Prediction
            if ldFlag or mcFlag or geFlag:
                landmarks = faceLandmarkDetector.predict(face)

                left_eye_X, left_eye_Y, right_eye_X, right_eye_Y, eye_width = int(landmarks[0]), int(landmarks[1]), int(landmarks[2]), int(landmarks[3]), int(eye_width)

                left_eye = face[left_eye_Y-eye_width:left_eye_Y+eye_width, left_eye_X-eye_width:left_eye_X+eye_width]
                right_eye = face[right_eye_Y-eye_width:right_eye_Y+eye_width, right_eye_X-eye_width:right_eye_X+eye_width]
            
            if hdFlag or mcFlag or geFlag:
                headpose_angles = headPoseEstimator.predict(face)

            if geFlag or mcFlag:
                gaze_vector = gazeEstimator.predict(left_eye, right_eye, headpose_angles)

            axisLength = 0.5 * face.shape[1]

            # Visualization
            if mcFlag or geFlag:

                gaze_arrow = (int(axisLength * gaze_vector[0][0]),int(axisLength * (-gaze_vector[0][1])))

                cv2.arrowedLine(img = batch,
                                pt1 = (coords[0]+left_eye_X, coords[1]+left_eye_Y),
                                pt2 = (coords[0]+left_eye_X + gaze_arrow[0], coords[1]+left_eye_Y + gaze_arrow[1]),
                                color = (0,255,255),
                                thickness = 2
                                )

                cv2.arrowedLine(img = batch,
                                pt1 = (coords[0]+right_eye_X, coords[1]+right_eye_Y),
                                pt2 = (coords[0]+right_eye_X + gaze_arrow[0],coords[1]+ right_eye_Y + gaze_arrow[1]),
                                color = (0,255,255),
                                thickness = 2
                                )
            elif hdFlag:

                # Visualization Head Pose
                yaw, pitch, roll = headpose_angles[0][0], headpose_angles[0][1], headpose_angles[0][2]

                sinYaw = sin(yaw * pi /180)
                sinPitch = sin(pitch * pi/180)
                sinRoll = sin(roll * pi/180)

                cosYaw = cos(yaw * pi /180)
                cosPitch = cos(pitch * pi/180)
                cosRoll = cos(roll * pi/180)

                centerOfFace_X, centerOfFace_Y = int((coords[0]+coords[2])/2), int((coords[1]+coords[3])/2)

                cv2.line(img = batch, 
                        pt1 = (centerOfFace_X, centerOfFace_Y),
                        pt2 = ((centerOfFace_X + int(axisLength*cosRoll * cosYaw + sinYaw * sinPitch * sinRoll)), (centerOfFace_Y + int(axisLength * cosPitch * sinRoll))),
                        color = (0,0,255),
                        thickness = 3)

                cv2.line(img = batch, 
                        pt1 = (centerOfFace_X, centerOfFace_Y),
                        pt2 = ((centerOfFace_X + int(axisLength*cosRoll * sinYaw + sinPitch * cosYaw * sinRoll)), (centerOfFace_Y + int(axisLength * cosPitch * sinRoll))),
                        color = (0,255,0),
                        thickness = 3)

                cv2.line(img = batch, 
                        pt1 = (centerOfFace_X, centerOfFace_Y),
                        pt2 = ((centerOfFace_X + int(axisLength * sinYaw * cosRoll)), (centerOfFace_Y + int(axisLength * sinPitch))),
                        color = (255,0,0),
                        thickness = 3)

            elif ldFlag:
                
                cv2.circle(face, (int(landmarks[0]), int(landmarks[1])), 10, (255,255,0), -1)
                cv2.circle(face, (int(landmarks[2]), int(landmarks[3])), 10, (255,255,0), -1)
                cv2.circle(face, (int(landmarks[4]), int(landmarks[5])), 10, (255,255,0), -1)
                cv2.circle(face, (int(landmarks[6]), int(landmarks[7])), 10, (255,255,0), -1)
                cv2.circle(face, (int(landmarks[8]), int(landmarks[9])), 10, (255,255,0), -1)


            elif fdFlag:
                batch = cv2.rectangle(batch, (coords[0],coords[1]), (coords[2],coords[3]), (0,255,0), thickness = 3)
  

            # If the Mouse Controller Flag is set then only move the mouse
            if mcFlag:
                
                controller.move(gaze_vector[0][0], gaze_vector[0][1])
        
        except Exception as e:
            cv2.putText(batch, "Landmarks or Face not Detected!", org = (10,50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, thickness = 2, fontScale = 1, color = (0,0,255))


        cv2.imshow("Output", batch)

        if key_pressed == 27:
           break

    feed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__": 

    args = build_parser().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(asctime)-15s %(message)s",
                    level=log.INFO, stream=sys.stdout)
    main(args)