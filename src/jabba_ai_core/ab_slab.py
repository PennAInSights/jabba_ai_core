import sys
import os
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
import SimpleITK as sitk
import numpy as np

import jabba_ai_core.core.image_predict as ip
import jabba_ai_core.core.image_preprocess as preprocess
import jabba_ai_core.core.sitk_utils as sitku
from jabba_ai_core.core.jabba_custom_objects import get_jabba_custom_objects

class AbSlab(ip.ImagePredictLikelihood):

    model=None

    def __init__(self,custom_objects={}):
        #print("custom objects length: " + str(len(custom_objects)))
        super().__init__(custom_objects=custom_objects)
        self.region=None

    def PrePredict(self):
        prep = preprocess.ImagePreprocess(self.input)
        prep.window=1800
        prep.level=400
        prep.slice_size=[256,256]
        prep.pixel_type=sitk.sitkFloat32
        prep.orientation='RAI'
        self.prepped=prep.Execute()

    def PostPredict(self):
        self.output = self.GetContinuousSlab()
    
    def GetExtractedImage(self):
        if self.output is None:
            self.Update()

        extracted = sitk.RegionOfInterest(self.prepped, self.output['Size'], self.output['Index'])
        og = sitku.resize_to_reference(extracted, self.input, interpolation="NearestNeighbor")
        return(og)

    # Get start and end index for continous section of positive weights
    # Account for possible instability at the ends with weights very close to zero
    #   weights - array of predicted values where >0 is abdomen, <0 is non-abdomen
    # Returns - starting and ending index of abdominal slab
    def GetContinuousSlab(self):

        #self.logger.debug("DEBUG: AbSlab.GetContinuousSlab()")

        if self.prepped is None:
            return None
        if self.model is None:
            return None
        if self.predictions is None:
            self.Update()

        weights = self.predictions[:,0]-0.5

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

        sz = list(self.prepped.GetSize())
        idx=[0]*self.prepped.GetDimension()
        
        idx[2] = startIdx
        sz[2] = endIdx-startIdx+1
        region = {'Index':idx, 'Size':sz}
        self.region=region
        return(self.region)


def main():

    my_parser = argparse.ArgumentParser(description='Identify abdominal slab')
    my_parser.add_argument('-m', '--model',  type=str, help='model weights', required=True)
    my_parser.add_argument('-i', '--input',  type=str, help='input nifti volume', required=True)
    my_parser.add_argument('-o', '--output', type=str, help="output nifti volume", required=False)
    my_parser.add_argument('-v', '--verbose', help="verbose output", action='store_true', default=False)
    args = my_parser.parse_args()

    AbSlab.model =  tf.keras.models.load_model(args.model, custom_objects=get_jabba_custom_objects())

    img = sitk.ReadImage(args.input, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        print("EXITING: Input image is not 3D")
        return(1)

    predictor = AbSlab(custom_objects=get_jabba_custom_objects())
    predictor.SetDebugOn()
    predictor.SetImage(img)
    predictor.Update()
    slabRegion = predictor.GetOutput()
    if args.verbose:
        print("slab_height_mm:"+'{:.4f}'.format(slabRegion['Size'][2]*img.GetSpacing()[2]))
        print("slab_start_index:"+str(slabRegion['Index'][2]))
        print("slab_end_index:"+str(slabRegion['Index'][2]+slabRegion['Size'][2]-1))
    if args.output:
        outImg = predictor.GetExtractedImage()
        sitk.WriteImage(outImg, args.output)
    if (not args.output) and (not args.verbose):
        print( str(slabRegion['Index'][2]) + ',' + str(slabRegion['Size'][2]) )

if __name__=="__main__":
    sys.exit(main())
