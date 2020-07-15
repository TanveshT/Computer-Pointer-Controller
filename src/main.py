from models import FaceDetector, FaceLandmarkDetector, GazeEstimator, HeadPoseEstimator
from argparse import ArgumentParser
from input_feeder import InputFeeder
import cv2
from math import sqrt
from mouse_controller import MouseController

def main():

    controller = MouseController('high','fast')

    feed = InputFeeder(input_type="cam", input_file="bin/demo.mp4")
    feed.load_data()

    faceDetector = FaceDetector(model_name = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001",conf=0.6)
    faceDetector.load_model()

    faceLandmarkDetector = FaceLandmarkDetector(model_name = "models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009")
    faceLandmarkDetector.load_model()

    headPoseEstimator = HeadPoseEstimator(model_name = "models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001")
    headPoseEstimator.load_model()

    gazeEstimator = GazeEstimator(model_name = "models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002")
    gazeEstimator.load_model()

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
    main()