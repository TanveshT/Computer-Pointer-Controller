from models import FaceDetector, FaceLandmarkDetector, GazeEstimator, HeadPoseEstimator
from argparse import ArgumentParser
from input_feeder import InputFeeder
import cv2
from math import sqrt

def main():

    feed = InputFeeder(input_type="cam")
    feed.load_data()

    faceDetector = FaceDetector(model_name = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001",conf=0.6)
    faceDetector.load_model()

    faceLandmarkDetector = FaceLandmarkDetector(model_name = "models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009")
    faceLandmarkDetector.load_model()

    headPoseEstimator = HeadPoseEstimator(model_name = "models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001")
    headPoseEstimator.load_model()

    gazeEstimator = GazeEstimator("models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002")
    gazeEstimator.load_model()

    for batch in feed.next_batch():

        key_pressed = cv2.waitKey(60)
        xmin, ymin, xmax, ymax = faceDetector.predict(batch)

        face = batch[ymin:ymax, xmin:xmax]

        #Calculation for eye width according to face dimensions
        eye_width = sqrt((face.shape[0]*face.shape[1])/100)

        batch = cv2.rectangle(batch, (xmin,ymin), (xmax,ymax), (0,255,0), thickness = 2)

        left_eye_X, left_eye_Y, right_eye_X, right_eye_Y = faceLandmarkDetector.predict(face)

        #face = cv2.circle(face, (int(left_eye_X), int(left_eye_Y)), radius = 2, color = (255,0,0), thickness = 2)
        #face = cv2.circle(face, (int(right_eye_X), int(right_eye_Y)), radius = 2, color = (0,255,0), thickness = 2)

        face = cv2.rectangle(face, (int(left_eye_X-eye_width), int(left_eye_Y-eye_width)), (int(left_eye_X+eye_width), int(left_eye_Y+eye_width)), color = (255,0,0), thickness = 1)
        face = cv2.rectangle(face, (int(right_eye_X-eye_width), int(right_eye_Y-eye_width)), (int(right_eye_X+eye_width), int(right_eye_Y+eye_width)), color = (255,0,0), thickness = 1)

        headpose_angles = headPoseEstimator.predict(face)

        cv2.imshow("Output", batch)


        if key_pressed == 27:
           break
    feed.close()

if __name__ == "__main__":
    main()