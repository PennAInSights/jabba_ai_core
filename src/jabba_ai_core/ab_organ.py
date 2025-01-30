import sys
import argparse
import os
import logging
import tensorflow as tf
from tensorflow.keras import backend as K
import SimpleITK as sitk
import numpy as np

import jabba_ai_core.core.image_predict as ip
import jabba_ai_core.core.image_preprocess as preprocess
import jabba_ai_core.core.sitk_utils as sitku
from jabba_ai_core.core.jabba_custom_objects import get_jabba_custom_objects

class AbOrgan(ip.ImagePredictImageSlices):

    def __init__(self,custom_objects={}):
        super().__init__(custom_objects=custom_objects)
        self.logger = logging.getLogger('AbOrgan')
        self.logger.setLevel(logging.DEBUG)
        self.region=None

    # This run immediately before using image for prediction
    # Result should be stored in self.prepped
    def PrePredict(self):
        self.logger.debug("AbOrgan.PrePredict()")
        prep = preprocess.ImagePreprocess(self.input)
        prep.window = 150
        prep.level = 30
        prep.slice_size = [256,256]
        prep.pixel_type = sitk.sitkFloat32
        prep.orientation = 'RAI'
        self.prepped=prep.Execute()

    # This runs immediately after using image for prediction
    def PostPredict(self):
        self.logger.debug("AbOrgan.PostPredict()")
        nvox = np.sum(self.predictions>0.5)
        self.logger.debug("Number of voxels: "+str(nvox))

        img = sitk.GetImageFromArray(np.squeeze(self.predictions))
        img.CopyInformation(self.prepped)
        resampled = sitku.resize_to_reference(img, self.input, interpolation="NearestNeighbor")
        self.output = resampled
        if nvox > 0:
            thresholded = sitk.BinaryThreshold(resampled, lowerThreshold=0.5, upperThreshold=1, insideValue=1, outsideValue=0)
            largest_comp = sitku.largest_connected_component(thresholded)
            self.output = largest_comp


def main():

    my_parser = argparse.ArgumentParser(description='Identify abdominal organs')
    my_parser.add_argument('-m', '--model',  type=str, help='model weights', required=True)
    my_parser.add_argument('-i', '--input',  type=str, help='input nifti volume', required=True)
    my_parser.add_argument('-o', '--output',  type=str, help='output nifti volume', required=True)   
    my_parser.add_argument('-v', '--verbose', help="verbose output", action='store_true', default=False)
    args = my_parser.parse_args()

    #tf.disable_eager_execution()
    # Suppress TensorFlow's warning messages
    tf.get_logger().setLevel('ERROR')
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    img = sitk.ReadImage(args.input)
    if img.GetDimension() != 3:
        print("EXITING: Input image is not 3D")
        return(1)

    predictor = AbOrgan(custom_objects=get_jabba_custom_objects())
    predictor.SetDebugOn()
    predictor.SetImage(img)
    predictor.LoadModel(args.model)
    predictor.Update()
    organ = predictor.GetOutput()

    if args.verbose:
         vol=np.sum(sitk.GetArrayFromImage(organ))*np.prod(organ.GetSpacing())
         print("volume_mm^3:"+'{:.4f}'.format(vol)) 

    sitk.WriteImage(organ, args.output)


if __name__=="__main__":
    sys.exit(main())
