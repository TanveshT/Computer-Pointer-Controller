from models import FaceDetector, FaceLandmarkDetector, GazeEstimator, HeadPoseEstimator
from argparse import ArgumentParser
from input_feeder import InputFeeder

def main():

    feed = InputFeeder(input_type="cam")
    feed.load_data()

    faceDetector = FaceDetector(model_name = "models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001",conf=0.6)
    faceDetector.load_model()

    faceLandmarkDetector = FaceLandmarkDetector()
    faceLandmarkDetector.load_model()

    gazeEstimator = GazeEstimator()
    gazeEstimator.load_model()

    headPoseEstimator = HeadPoseEstimator()
    headPoseEstimator.load_model()

    for batch in feed.next_batch():
        print(faceDetector.predict(batch))

    feed.close()

if __name__ == "__main__":
    main()