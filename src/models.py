from openvino.inference_engine import IECore
import cv2
import numpy as np

class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, conf = 0.5):
        '''
        Initializes the model by taking in the model path
        '''
        self.model_structure = model_name + '.xml'
        self.model_weights = model_name + '.bin'
        self.device = device
        self.exec_net = None
        self.model = None
        self.confidence = conf
            

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        self.model = core.read_network(model = self.model_structure, weights = self.model_weights)
        self.exec_net = core.load_network(network = self.model, device_name = self.device)

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def predict(self, image):
        '''
        Description: This method is meant for running predictions on the input image.
        params:
            image: the original image
        returns:
            xmin, ymin, xmax, ymax: Face Coordinates
        '''
        processed_image = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:processed_image})

        return self.preprocess_output(outputs)

    def check_model(self):
        ''''''

    def preprocess_input(self, image):
        '''
        Description: This method is meant for preprocessing the image to the required model dimensions.
        params:
            image: the original image
        returns:
            processed_image: Image converted to the required model shape
        '''
        self.image = image
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Description: This method is meant for running Preprocessing on the model outputs
        params:
            outputs: Model Output Provdided here and prorcessed as per the requirements
        returns:
            xmin, ymin, xmax, ymax: Face Coordinates
        '''
        boxes = outputs[self.output_name][0][0]
        xmin = 0; ymin = 0; xmax = 0; ymax = 0
        for box in boxes:
            if box[2] > self.confidence:
                xmin = int(box[3] * self.image.shape[1])
                ymin = int(box[4] * self.image.shape[0])
                xmax = int(box[5] * self.image.shape[1])
                ymax = int(box[6] * self.image.shape[0])

        return [xmin, ymin, xmax, ymax]

class FaceLandmarkDetector:
    '''
    Class for the Face Landmark Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initializes the model by taking in the model path
        '''
        self.model_structure = model_name + '.xml'
        self.model_weights = model_name + '.bin'
        self.device = device
        self.exec_net = None
        self.model = None
            

    def load_model(self):
        '''
        Description: This method is for loading the model to the device specified by the user.
        '''
        core = IECore()
        self.model = core.read_network(model = self.model_structure, weights = self.model_weights)
        self.exec_net = core.load_network(network = self.model, device_name = self.device)

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def predict(self, image):
        '''
        Description: This method is meant for running predictions on the input image.
        params:
            image: the original image
        returns:
            #TODO
        '''
        
        processed_image = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:processed_image})
        return self.preprocess_output(outputs)


    def check_model(self):
        ''''''

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        self.image = image
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        heatmap =  outputs[self.output_name]
        x0, y0, x1, y1 = heatmap[0][0][0][0], heatmap[0][1][0][0], heatmap[0][2][0][0], heatmap[0][3][0][0]
        x0, y0, x1, y1 = x0*self.image.shape[1], y0*self.image.shape[0], x1*self.image.shape[1], y1*self.image.shape[0]

        return x0, y0, x1, y1
        
class HeadPoseEstimator:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initializes the model by taking in the model path
        '''
        self.model_structure = model_name + '.xml'
        self.model_weights = model_name + '.bin'
        self.device = device
        self.exec_net = None
      

    def load_model(self):
        '''
        Description: This method is for loading the model to the device specified by the user.
        '''
        core = IECore()
        self.model = core.read_network(model = self.model_structure, weights = self.model_weights)
        self.exec_net = core.load_network(network = self.model, device_name = self.device)
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def predict(self, image):
        '''
        Description: 
            This method is meant for running predictions on the input image.
        Params:
            image: The input image
        Returns:
            The array received by the method preprocess_outputs() i.e headpose angles
        '''
        processed_image = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:processed_image})
        return self.preprocess_output(outputs)

    def check_model(self):
        ''''''

    def preprocess_input(self, image):
        '''
        Description:
            Preprocess inputs according to the model input dimensions
        Params:
            image: The input image from capture feed
        Returns:
            processed_image: Image with dimensions that matches model input dimensions
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Description:
            Preprocess Outputs before passing it to next model
        Params: 
            outputs: Output received by traversing the model
        Returns:
            Array: An array with headpose angles "yaw", "pitch", "roll"
        '''
        yaw = outputs["angle_y_fc"][0][0]
        pitch = outputs["angle_p_fc"][0][0]
        roll = outputs["angle_r_fc"][0][0]

        return np.array([[yaw, pitch, roll]])

class GazeEstimator:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        Initializes the model by taking in the model path
        '''
        self.model_structure = model_name + '.xml'
        self.model_weights = model_name + '.bin'
        self.device = device
        self.exec_net = None
        self.model = None
            

    def load_model(self):
        '''
        Description:
            This method is for loading the model to the device specified by the user.
        '''
        core = IECore()
        self.model = core.read_network(model = self.model_structure, weights = self.model_weights)
        self.exec_net = core.load_network(network = self.model, device_name = self.device)

        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.input_shape = self.model.inputs["left_eye_image"].shape

    def predict(self, left_eye, right_eye, headpose_angles):
        '''
        Description: 
            This method is meant for running predictions on the input image.
        Params:
            left_eye: The left eye extracted from face
            right_eye: The right eye image extracted from face
            headpose_angles: yaw, pitch, roll angles in degrees fetched from Head Pose Estimation Model
        '''

        processed_left_eye = self.preprocess_input(left_eye)
        processed_right_eye = self.preprocess_input(right_eye)
        input_dict = {"left_eye_image": processed_left_eye, "right_eye_image": processed_right_eye, "head_pose_angles": headpose_angles}

        outputs = self.exec_net.infer(input_dict)
        return self.preprocess_output(outputs)

    def check_model(self):
        ''''''

    def preprocess_input(self, image):
        '''
        Description:
            Preprocess inputs according to the model input dimensions
        Params:
            image: The RIGHT or LEFT eye for preprocessing
        Returns:
            processed_image: Returns the eye image with model dimensions
        '''

        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Description:
            Preprocess the outputs and send the gaze_vector
        Params:
            outputs: the ouptus receivvec from the output
        Returns:
            gaze_vector: 
        '''
        
        gaze_vector = outputs[self.output_name]
        print(gaze_vector)
        return gaze_vector
