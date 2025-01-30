#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
import SimpleITK as sitk
import numpy as np
import jabba_ai_core.core.image_predict as ip
import jabba_ai_core.core.image_preprocess as preprocess
import logging
import os
from jabba_ai_core.core.jabba_custom_objects import get_jabba_custom_objects

class AbContrast(ip.ImagePredictLikelihood):

    def __init__(self, custom_objects={}):
        super().__init__(custom_objects=custom_objects)
        self.region=None
        self.logger = logging.getLogger("AbContrast")
        self.logger.setLevel(logging.DEBUG)

    def PrePredict(self):
        self.logger.debug("AbContrast.PrePredict()")
        prep = preprocess.ImagePreprocess(self.input)
        prep.window = 150
        prep.level = 30
        prep.slice_size = [256,256]
        prep.pixel_type = sitk.sitkFloat32
        prep.orientation = 'RAI'
        self.prepped=prep.Execute()

    def PostPredict(self):
        self.output = self.predictions[:,0]

    def IsContrastEnhanced(self):
        self.logger.debug("AbContrast.IsContrastEnhanced()")
        if self.predictions is None:
            self.Update()

        meanChance = np.mean(self.output)
        outBool = False
        if meanChance > 0.5:
            outBool = True 

        return(outBool)




# Get start and end index for continous section of positive weights
# Account for possible instability at the ends with weights very close to zero
#   weights - array of predicted values where >0 is abdomen, <0 is non-abdomen
# Returns - starting and ending index of abdominal slab
def getContinuousSlab(weights):
    cumsum = 0
    highsum = 0
    startIdx = 0
    endIdx = 0
    currentIdx = 0

    for idx, val in enumerate(weights):
        # Start if positive weight
        # Record cumulative sum of positive weights
        # Reset if a 'more negative' weight than cumsum is found
        if cumsum+val > 0:
            cumsum += val
        else:
            cumsum = 0
            currentIdx = idx+1

        if cumsum > highsum:
            startIdx = currentIdx
            endIdx = idx
            highsum = cumsum

    return (startIdx, endIdx)

# NOTE: this implemention sets level -= 0.5 and window -= 1
def windowLevelArray(img, level, width, rescaleSlope=1, rescaleIntercept=0, outMin=0, outMax=255):
    img = img.astype(np.float, copy=False)
    img = img * rescaleSlope + rescaleIntercept

    img =  np.piecewise(img,
        [img <= (level - 0.5 - (width-1)/2),
        img > (level - 0.5 + (width-1)/2)],
        [outMin, outMax, lambda img: ((img - (level - 0.5))/(width-1) + 0.5)*(outMax-outMin)])
    #img = img.astype(np.uint8, copy=False)
    return img

def getSliceWeights(model, stack, zBatchSize = 32):

    imgShape = (model.input.shape[1], model.input.shape[2])
    nstack = arrayResize(stack, imgShape, method=cv2.INTER_CUBIC)
    nstack = nstack[:, :, :, np.newaxis]
    nstack = windowLevelArray(nstack, level=30, width=150)
    #nstack = nstack.astype(np.uint8,copy=False) # Testing
    nstack = nstack.astype("float")

    chanceContrastList = np.array([])
    for zIndex in range(0, nstack.shape[0], zBatchSize):
        maxZ = (min(zIndex+zBatchSize, nstack.shape[0]))
        sample = nstack[zIndex:maxZ, :, :, :]
        pred= model.predict(sample)[:,0]
        chanceContrastList = np.concatenate([chanceContrastList, pred])

    return(chanceContrastList)



def main():

    my_parser = argparse.ArgumentParser(description='Identify contrast enhanced volumes')
    my_parser.add_argument('-m', '--model',  type=str, help='model weights', required=True)
    my_parser.add_argument('-i', '--input',  type=str, help='input nifti volume', required=True)
    my_parser.add_argument('-v', '--verbose', dest="verbosity", help="verbose output", action='store_true', default=False)
    args = my_parser.parse_args()

    #tf.disable_eager_execution()
    # Suppress TensorFlow's warning messages
    tf.get_logger().setLevel('ERROR')
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    img = sitk.ReadImage(args.input, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        print("EXITING: Input image is not 3D")
        return(1)
    
    # FIXME - check FOV in z-direction

    #modelSlice = tensorflow.keras.models.load_model(args.model)
    #contrastChances = getSliceWeights( modelSlice, itkImg )
    #if args.verbosity:
    #    print(contrastChances)
    #print( "Chance of contrast: " + str(np.mean(contrastChances)))

    #print("Try with class")
    predictor = AbContrast(custom_objects=get_jabba_custom_objects())
    predictor.SetDebugOn()
    predictor.SetImage(img)
    predictor.LoadModel(args.model)
    predictor.Update()
    chances = predictor.GetOutput()
    if args.verbosity:
        print(chances)
        print( "Chance of contrast: " + '{:.4f}'.format( np.mean(chances) ) )
    else:
        print( '{:.4f}'.format( np.mean(chances) ) )


if __name__=="__main__":
    #print("Running abSlab")
    sys.exit(main())
