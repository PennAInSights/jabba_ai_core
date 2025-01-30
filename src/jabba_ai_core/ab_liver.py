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
from jabba_ai_core.ab_organ import AbOrgan



class AbLiver(AbOrgan):

    model = None

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('AbLiver')
        self.logger.setLevel(logging.DEBUG)
        self.region=None

    def HasModel(self):
        if AbLiver.model is None:
            return(False)
        else:
            return(True)
        

def main():

    my_parser = argparse.ArgumentParser(description='Identify liver')
    my_parser.add_argument('-m', '--model',  type=str, help='model weights', required=True)
    my_parser.add_argument('-i', '--input',  type=str, help='input nifti volume', required=True)
    my_parser.add_argument('-o', '--output',  type=str, help='output nifti volume', required=True)   
    my_parser.add_argument('-v', '--verbose', help="verbose output", action='store_true', default=False)
    args = my_parser.parse_args()

    #tf.disable_eager_execution()
    # Suppress TensorFlow's warning messages
    tf.get_logger().setLevel('ERROR')
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    AbLiver.model =  tf.keras.models.load_model(args.model, custom_objects=get_jabba_custom_objects())

    img = sitk.ReadImage(args.input)
    if img.GetDimension() != 3:
        print("EXITING: Input image is not 3D")
        return(1)

    predictor = AbLiver()
    predictor.SetDebugOn()
    predictor.SetImage(img)
    predictor.SetModel(AbLiver.model)
    predictor.Update()
    organ = predictor.GetOutput()

    if args.verbose:
         vol=np.sum(sitk.GetArrayFromImage(organ))*np.prod(organ.GetSpacing())
         print("volume_mm^3:"+'{:.4f}'.format(vol)) 

    sitk.WriteImage(organ, args.output)


if __name__=="__main__":
    sys.exit(main())
