import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import logging
import os
import torch
import monai
from monai.inferers import sliding_window_inference
import datetime
import jabba_ai_core.core.sitk_utils as sitku


# Base class for using images to make deep learning based predictions
class ImagePredict():

    def __init__(self, custom_objects={}):
        self.model=None
        self.predictions=None
        self.debug=False
        self.prepped=None        # Image prepped for prediction
        self.output=None       
        self.batchSize=12
        self.input=None           # Original input image
        self.pipeline=None
        self.custom_objects=custom_objects
        self._logger = logging.getLogger('ImagePredict')
        self._logger.setLevel(logging.DEBUG)
        self.region=None
        self.modelType = 'tensorflow'
        self._device = 'cpu'
        self.orientation='RAI'
        self.time=False
        self.runtime="NA"
        self.name="ImagePredict"

    def __str__(self):
        msg=self.__repr__()
        msg=msg+"\n  Image: "+self.image.__repr__()
        msg=msg+"\n  Model: "+self.model.__repr__()
        return(msg)
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        if value == 'cpu':
            self._device = 'cpu'
        elif value == 'cuda':
            self._device = 'cuda'
        else:
            raise ValueError(f'Invalid device: {value}')    
        
    @property
    def logger(self):
        return self._logger
    
    @logger.setter
    def logger(self, x):
        if not isinstance(x, logging.Logger):
            raise TypeError("Input logger must be a logging.Logger")
        self._logger = x

    def SetBatchSize(self, bSize):
        self.batchSize = bSize

    def SetDebugOn(self):
        self.debug = True
        self.logger.setLevel(logging.DEBUG)

    def SetDebugOff(self):
        self.debug = False
        self.logger.setLevel(logging.INFO)

    def SetDevice(self, device):
        self.device = device

    # Set input image and check against model if present
    def SetImage(self, x):
        self.logger.debug("ImagePredict.SetImage()")
        self.predictions=None
        template=None

        if not isinstance(x, sitk.Image):
            self.logger.error("Input image must be an sitk.Image")
            return False

        if x.GetDimension() != 3:
            self.logger.error("Input image must be 3D")
            return False           

        self.input=x

    # Set model and check against image if present
    def SetModel(self, x):
        self.logger.debug("ImagePredict.SetModel()")
        self.model=x

    def HasModel(self):
        return(not self.model is None)

    def LoadModel(self, modelName):
        self.logger.debug("ImagePredict.LoadModel()")
        if os.path.isdir(modelName):
            self.modelType='tensorflow'
        else:
            split_ext = os.path.splitext(modelName)
            if split_ext[1]=='.h5':
                self.modelType='tensorflow'
            elif split_ext[1]=='.pt':
                self.modelType='torch'
            else:
                self.logger.error("Unknown type for model file")
                raise RuntimeError("Model file has unknown type")

        if self.modelType=='tensorflow':
            self.LoadTensorflowModel(modelName)
        if self.modelType=='torch':
            self.LoadTorchModel(modelName)

        self.logger.info("ModelType: "+self.modelType)

    def LoadTensorflowModel(self, modelName):
        self.logger.debug("ImagePredict.LoadTensorflowModel()")
        model = tf.keras.models.load_model(modelName, custom_objects=self.custom_objects)
        self.SetModel(model)

    def LoadTorchModel(self, modelName):
        self.logger.debug("ImagePredict.LoadTorchModel()")
        try:
            model = torch.jit.load(modelName, self.device)
            self.SetModel(model)
        except: 
            self.logger.error("Failed to load torch model")

        
    # Set number of slices to pass to model at once
    def SetBatchSize(self, x):
        self.logger.debug("ImagePredict.SetBatchSize()")
        self.batchSize=x

    def GetModelInputShape(self):
        if self.model is None:
            return None

        if self.modelType=="tensorflow":
            return(self.model.input.shape)

        if self.modelType=='torch':
            return( (None,256,256) )

    # Compare image and model parameters
    def ValidateInput(self):
        self.logger.debug("ImagePredict.ValidateInput()")

        if self.input is None:
            raise RuntimeError("Missing input image")

        if self.model is None:
            raise RuntimeError("Missing AI model")

        # check data type agreement
        #template = itk.template(self.input)
        #pixelType = template[1][0]

        # Check slice size
        #imageSliceSize = [self.input.shape[1], self.input.shape[2]]
        imageSliceSize = [self.prepped.GetSize()[1], self.prepped.GetSize()[0]]

        inShape = self.GetModelInputShape()
        modelSliceSize = [int(inShape[1]), int(inShape[2])]
        if not imageSliceSize==modelSliceSize:
            self.logger.error("Image and model are not the same size")
            raise RuntimeError("Image and model are not the same size")

        # Check datatype - necessary?
        # model_dtype = self.model.input.dtype
        # print(model_dtype)

    # Unused (will use in itk==5.3.0 when released)
    def GenerateData(self, py_image_filter):

        input = py_image_filter.GetInput()
        output = py_image_filter.GetOutput()
        output.Graft(input)

    def SetRegion(self, region):
        self.region = region

    def PrePredict(self):
        self.logger.debug("ImagePredict.PrePredict()")
        self.prepped = self.input

    def Predict(self):
        self.logger.debug("ImagePredict.Predict()")
        if self.modelType=='tensorflow':
            self.TensorflowPredict()
        elif self.modelType=='torch':
            self.TorchPredict()
        else:
            self.logger.error("Invalid model type")
            raise RuntimeError("Invalid model type")

    # Apply model to prepped input
    def TensorflowPredict(self):
        self.logger.debug("ImagePredict.TensorflowPredict()")
        nstack = sitk.GetArrayFromImage(self.prepped)

        nstack = nstack[:, :, :, np.newaxis]
        outshape = np.asarray(self.model.output_shape)
        outshape[0] = nstack.shape[0]
        self.predictions = np.zeros(outshape, dtype=float)

        regionStart=0
        regionEnd=nstack.shape[0]
        if not self.region is None:
            #regionStart = self.region.GetIndex()[2]
            #regionEnd = regionStart + self.region.GetSize()[2]
            regionStart = self.region['Index'][2]
            regionEnd = regionStart + self.region['Size'][2]           

        self.logger.debug("Predicting in slice range: "+str(regionStart)+' -> '+str(regionEnd-1))

        for zIndex in range(regionStart, regionEnd, self.batchSize):
            zMax = (min(zIndex+self.batchSize, regionEnd))
            sample = nstack[zIndex:zMax, :, :, :]
            nChannels = self.GetModelInputShape()[3]

            #self.logger.info("Predicting slice: "+str(zIndex)+' -> '+str(zMax-1))
            #print( sample.shape )
            # For more than 1 channel, usually means RGB images were used to train
            if nChannels > 1:
                sample = np.squeeze(np.stack([ sample for _ in range(nChannels)],axis=3))

            # Account for single slice case
            if len(sample.shape)==3:
                sample = np.expand_dims(sample, axis=0)
            #print( sample.shape )

            ten = tf.convert_to_tensor(sample)

            #FIXME - there must be a generic way to do this
            if len(outshape)==2:
                self.predictions[zIndex:zMax, :] = self.model.predict_on_batch(ten)
            elif len(outshape)==3:
                self.predictions[zIndex:zMax, :, :] = self.model.predict_on_batch(ten)
            elif len(outshape)==4:
                self.predictions[zIndex:zMax, :, :, :] = self.model.predict_on_batch(ten)

    def TorchPredict(self):
        self.logger.debug("ImagePredict.TorchPredict()")
        #print(summary(self.model))
        nstack = sitk.GetArrayFromImage(self.prepped)

        #outshape[0] = nstack.shape[0]
        #outshape = np.asarray(self.input.shape)
        outshape = (nstack.shape[0], 256, 256)
        self.predictions = np.zeros(outshape, dtype=float)

        regionStart=0
        regionEnd=nstack.shape[0]
        if not self.region is None:
            #regionStart = self.region.GetIndex()[2]
            #regionEnd = regionStart + self.region.GetSize()[2]
            regionStart = self.region['Index'][2]
            regionEnd = regionStart + self.region['Size'][2] 
        self.logger.debug("Predicting in slice range: "+str(regionStart)+' -> '+str(regionEnd-1))

        nstack = nstack[regionStart:regionEnd, :, :, np.newaxis]
        ten = torch.from_numpy(nstack).permute(0,3,1,2)

        # FIXME - can we get this from the model?
        roi_size = (256, 256)
        sw_batch_size = 1
        test_outputs = sliding_window_inference(ten.to(self.device), roi_size, sw_batch_size, self.model)
        test_outputs = torch.argmax(test_outputs, dim=1).detach().cpu()
        test_outputs_np = test_outputs.detach().numpy().astype('uint8')

        self.predictions[regionStart:regionEnd,:,:] = test_outputs_np

        preBuffer = np.unique( self.predictions[ 0:(regionStart-1), :, :])
        buffer = np.unique( self.predictions[regionStart:regionEnd, :, :])
        postBuffer = np.unique( self.predictions[(regionEnd+1):(outshape[0]-1) ])

    def PostPredict(self):
        self.logger.debug("ImagePredict.PostPredict()")
        #labels = np.unique( self.predictions ).sort()
        #print("Predicted labels")
        #print(labels)

    def Update(self):
        self.logger.debug("ImagePredict.Update()")
        start=None
        if self.time:
            start = datetime.datetime.now()

        self.PrePredict()
        self.ValidateInput()
        self.Predict()
        self.PostPredict()

        if self.time:
            end = datetime.datetime.now()
            self.logger.info(self.name + " prediction time: "+str(end-start))

    def GetOutput(self):
        if self.output is None:
            self.Update()
        return(self.output)

# Class for applying a prediction on a slice-by-slice basis
class ImagePredictImageSlices(ImagePredict):
    def __init__(self,custom_objects={}):
        super().__init__(custom_objects)

    # Apply preprocessing - Define in derived class
    def PrePredict(self):
        self.logger.debug("ImagePredictImageSlices.PrePredict()")
        self.input = self.image

    #def TorchPredict(self):
    #    self.logger.debug("ImagePredictImageSlices.TorchPredict")
    #    raise RuntimeError("TorchPredict() is not implemented")

    # Apply postprocessing - Define in derived class
    def PostPredict(self):
        self.logger.debug("ImagePredictImageSlices.PostPredict()")
        self.output = sitk.GetImageFromArray(np.squeeze(self.predictions))
        self.output.CopyInformation(self.input)
        self.output = self.predictions
        
class ImagePredictLikelihood(ImagePredict):

    def __init__(self, custom_objects={}):
        super().__init__(custom_objects=custom_objects)

    def UpdateOld(self):
        self.logger.debug("ImagePredictLikelihood.Update()")

        self.PreUpdate()

        nstack = itk.GetArrayViewFromImage(self.input)
        nstack = nstack[:, :, :, np.newaxis]

        likelihood = np.array([])
        for zIndex in range(0, nstack.shape[0], self.batchSize):
            maxZ = (min(zIndex+self.batchSize, nstack.shape[0]))
            sample = nstack[zIndex:maxZ, :, :, :]
            #predValues = self.model.predict(tf.convert_to_tensor(sample))[:,0]
            nChannels = self.model.input.shape[3]
            if nChannels > 1:
                self.logger.debug("Converting scalar sample to grayscale RGB")
                sample = np.stack([sample for _ in range(nChannels)], axis=3)

            predValues = self.model.predict(tf.convert_to_tensor(sample))[:,0]

            #predValuesB = self.model.predict(sample)[:,0]
            #print("Prediction SSD="+str(np.sum( (predValues-predValuesB)**2)))

            likelihood = np.concatenate([likelihood, predValues])


        self.predictions = likelihood
