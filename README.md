# Computer Pointer Controller

This project is made using the Intel's OpenVINO toolkit. This project enables the user to move the mouse pointer according to the gaze of the user.

## Project Set Up and Installation

Requirements:
* Intel's OpenVINO toolkit

After installing OpenVINO 
```bash
source /opt/intel/openvino/bin/setupvars.sh -pyver {PYTHON_VERSION}
```

## Demo

### To run demo video

```bash
python main.py -input_type video -input_stream bin/demo.mp4 
```

### To run demo cam

```bash
python main.py -input_type cam
```

## Documentation

### Command-Line Arguments

```bash
usage: main.py [-h] [-input_stream INPUT_STREAM]
               [-device {CPU,GPU,HETERO:FPGA,CPU,HETERO:MYRIAD,CPU}]
               [-prob_threshold PROB_THRESHOLD] -input_type {cam,video,image}

optional arguments:
  -h, --help            show this help message and exit
  -input_stream INPUT_STREAM
                        Give path of input stream if input_type is 'video' or
                        'image'
  -device {CPU,GPU,HETERO:FPGA,CPU,HETERO:MYRIAD,CPU}
                        State the device on which inference should happen
  -prob_threshold PROB_THRESHOLD
                        Probability threshold for face detections filtering

required arguments:
  -input_type {cam,video,image}
                        Give the type of input stream
```

## Results

### Loading times of Models

**CPU used: Intel(R) Core(TM) i3-6006U CPU @ 2.00GHz**

|                                                     | FP32 (in ms)        |     FP16 (in ms)    | FP16-INT8 (in ms)   |
|-----------------------------------------------------|---------------------|:-------------------:|---------------------|
| Face Detection Model (Same Model for all scenarios) | 0.2433936595916748  | 0.26357316970825195 | 0.2504093647003174  |
| Face Landmark Detection Model                       | 0.1449739933013916  | 0.6543600559234619  | 0.14608454704284668 |
| Headpose Detection Model                            | 0.14208269119262695 | 0.6201651096343994  | 0.271759033203125   |
| Gaze Estimation Model                               | 0.21468305587768555 | 0.3256380558013916  | 0.3227040767669678  |

## Edge Cases

* The face which is closest to camera will be taken into consideration and the gaze of that particular face will be calculated.
* Even if there is no one in the frame the application will be running.