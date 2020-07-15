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

```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
|                                                     | FP32 (in ms)        |     FP16 (in ms)    | FP16-INT8 (in ms)   |
|-----------------------------------------------------|---------------------|:-------------------:|---------------------|
| Face Detection Model (Same Model for all scenarios) | 0.2433936595916748  | 0.26357316970825195 | 0.2504093647003174  |
| Face Landmark Detection Model                       | 0.1449739933013916  | 0.6543600559234619  | 0.14608454704284668 |
| Headpose Detection Model                            | 0.14208269119262695 | 0.6201651096343994  | 0.271759033203125   |
| Gaze Estimation Model                               | 0.21468305587768555 | 0.3256380558013916  | 0.3227040767669678  |

## Edge Cases

* The face which is closest to camera will be taken into consideration and the gaze of that particular face will be calculated.
* Even if there is no one in the frame the application will be running.