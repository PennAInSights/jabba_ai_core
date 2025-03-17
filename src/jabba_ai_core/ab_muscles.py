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

import logging


class AbMuscles(ip.ImagePredictImageSlices):

    def __init__(self,custom_objects={}):
        super().__init__(custom_objects=custom_objects)
        self.logger = logging.getLogger('AbMuscles')
        self.logger.setLevel(logging.DEBUG)
        self.region=None

        self._fat_threshold=-10
        self._minimum_area=10
        self._intra_muscular_fat=None


    @property 
    def fat_threshold(self):
        return self._fat_threshold
    
    @fat_threshold.setter
    def fat_threshold(self, x):
        if not isinstance(x, (int, float)):
            raise TypeError("Input fat_threshold must be a number")
        self._fat_threshold = x

    @property
    def minimum_area(self):
        return self._minimum_area
    
    @minimum_area.setter
    def minimum_area(self, x):
        if not isinstance(x, (int,float)):
            raise TypeError("Input minimum_area must be an number")
        if x < 0:
            raise ValueError("Input minimum_area must be >= 0")
        self._minimum_area = x

    @property
    def intra_muscular_fat(self):
        return self._intra_muscular_fat
    
    @intra_muscular_fat.setter
    def intra_muscular_fat(self, x):
        raise TypeError("intra_muscular_fat is a read-only property")
    
    def PrePredict(self):
        self.logger.debug("AbMuscles.PrePredict()")
        prep = preprocess.ImagePreprocess(self.input)
        prep.window = 400
        prep.level = 40
        prep.slice_size = [256,256]
        prep.pixel_type = sitk.sitkFloat32
        prep.orientation = 'ARI'
        self.prepped=prep.Execute()

    def RemoveIntraMuscularFat(self, img, muscles):
        self.logger.debug("AbMuscles.RemoveFat()")
        imf = (img < self.fat_threshold) & (muscles > 0)
        if self.minimum_area > 0:
            imf_labeled = sitk.ConnectedComponent(imf)
            imf_labeled = sitk.RelabelComponent(imf_labeled, minimumObjectSize=self.minimum_area)
        imf = imf_labeled > 0
        imf_inverse = img==0
        imf = imf * self.output 
        self._intra_muscular_fat = sitk.Cast(imf, sitk.sitkUInt8)
        self.output = self.output - imf

    def GetIntraMusclularFat(self, img, muscles):

        fat_mask = self.input < self.fat_threshold
        muscle_mask = self.output > 0
        imf = fat_mask & muscle_mask

        imf = (img < self.fat_threshold) & (muscles > 0)
        if self.minimum_area > 0:
            imf_labeled = sitk.ConnectedComponent(imf)
            imf_labeled = sitk.RelabelComponent(imf_labeled, minimumObjectSize=self.minimum_area)
        imf = imf_labeled > 0
        imf_inverse = img==0
        imf = imf * self.output 
        self._intra_muscular_fat = sitk.Cast(imf, sitk.sitkUInt8)


        #fatstack = (anat < fat_thresh) & (musc > 0) # (CT voxel less than fat threshold) and (within mask)
        #if minarea:
        #    fatstack=skimage.morphology.remove_small_objects(fatstack,minarea,connectivity=2)
        #fatstack = fatstack.astype(int)
        #return fatstack

    def PostPredict(self):
        self.logger.debug("AbMuscles.PostPredict()")
        img = sitk.GetImageFromArray(np.squeeze(self.predictions))
        img.CopyInformation(self.prepped)
        img = sitk.Cast(img, sitk.sitkUInt8)
        connected = sitku.largest_connected_component_per_label(img)
        self.output = sitku.resize_to_reference(connected, self.input, interpolation="NearestNeighbor")
        #self.RemoveIntraMuscularFat(self.input, self.output)
        self.GetIntraMusclularFat(self.input, self.output)


def main():

    my_parser = argparse.ArgumentParser(description='Identify contrast enhanced volumes')
    my_parser.add_argument('-m', '--model',  type=str, help='model weights', required=True)
    my_parser.add_argument('-i', '--input',  type=str, help='input nifti volume', required=True)
    my_parser.add_argument('-z', nargs=2, help='first and last slice in abdominal slab', required=False)
    my_parser.add_argument('-o', '--output',  nargs='+', type=str, help='output nifti volume', required=True)   
    my_parser.add_argument('-v', '--verbose', dest="verbosity", help="verbose output", action='store_true', default=False)
    args = my_parser.parse_args()
    if args.verbosity:
        print(args)

    #tf.disable_eager_execution()
    # Suppress TensorFlow's warning messages
    tf.get_logger().setLevel('ERROR')
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    img = sitk.ReadImage(args.input, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        print("EXITING: Input image is not 3D")
        return(1)
    size = list(img.GetSize())
    index = [0] * img.GetDimension()

    predictor = AbMuscles()
    predictor.SetDebugOn()
    predictor.SetImage(img)
    predictor.LoadModel(args.model)
    if args.z:
        index[2] = int(args.z[0])
        size[2] = int(args.z[1])
    predictor.SetRegion( {'Index':index, 'Size':size} )
    predictor.Update()
    muscles = predictor.GetOutput()

    sitk.WriteImage(muscles, args.output[0])
    if len(args.output) > 1:
        sitk.WriteImage(predictor.intra_muscular_fat, args.output[1])

if __name__=="__main__":
    sys.exit(main())
