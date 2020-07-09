from openvino.inference_engine import IECore
import cv2

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

        '''
        processed_image = self.preprocess_input(image)
        outputs = self.exec_net.infer({self.input_name:processed_image})
        xmin, ymin, xmax, ymax = self.preprocess_output(outputs)

        return xmin, ymin, xmax, ymax

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
        boxes = outputs[self.output_name][0][0]
        xmin = 0; ymin = 0; xmax = 0; ymax = 0
        for box in boxes:
            if box[2] > self.confidence:
                xmin = int(box[3] * self.image.shape[1])
                ymin = int(box[4] * self.image.shape[0])
                xmax = int(box[5] * self.image.shape[1])
                ymax = int(box[6] * self.image.shape[0])

        return xmin, ymin, xmax, ymax

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
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.inputs[self.output_name].shape
        self.exec_net = None
        self.model = None
            

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        self.model = core.read_network(model = self.model_structure, weights = self.model_weights)
        self.exec_net = core.load_network(network = self.model, device_name = self.device)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError

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
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.inputs[self.output_name].shape
        self.exec_net = None
        self.model = None
            

    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        self.model = core.read_network(model = self.model_structure, weights = self.model_weights)
        self.exec_net = core.load_network(network = self.model, device_name = self.device)

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError

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
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        core = IECore()
        self.model = core.read_network(model = self.model_structure, weights = self.model_weights)
        self.exec_net = core.load_network(network = self.model, device_name = self.device)
        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.inputs[self.output_name].shape

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        processed_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        processed_image = processed_image.transpose((2,0,1))
        processed_image = processed_image.reshape(1, *processed_image.shape)
        
        return processed_image

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
