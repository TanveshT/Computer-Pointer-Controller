from openvino.inference_engine import IECore
import cv2
import numpy as np

class Model:

    def __init__(self, model_name, device='CPU', extensions=None, conf = 0.5):
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