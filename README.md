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
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Edge Cases

* The face which is closest to camera will be taken into consideration and the gaze of that particular face will be calculated.
* Even if there is no one in the frame the application will be running.