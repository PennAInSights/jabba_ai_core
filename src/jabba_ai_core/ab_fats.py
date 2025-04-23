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

class AbFats(ip.ImagePredictImageSlices):

    model = None

    def __init__(self, region=None, custom_objects={}):
        super().__init__(custom_objects=custom_objects)

    def SetRegion(self,region):
        self.region=region

    def PrePredict(self):
        self.logger.debug("AbFat.PrePredict()")
        prep = preprocess.ImagePreprocess(self.input)
        prep.window=400
        prep.level=50
        prep.slice_size=[256,256]
        prep.pixel_type=sitk.sitkFloat32
        prep.orientation='RAI'
        self.prepped=prep.Execute()

    def PostPredict(self):
        self.logger.debug("AbFat.PostPredict()")

        img = sitk.GetImageFromArray(np.squeeze(self.predictions))
        img.CopyInformation(self.prepped)
        resized = sitku.resize_to_reference(img, self.input, interpolation="NearestNeighbor")

        #self.logger.debug("AbFat.PostPredict() - check 1")
        
        mask_inner = resized > 0.5
        thresh_outer = self.input > -300

        #self.logger.debug("AbFat.PostPredict() - check 2")

        mask_outer = sitku.largest_connected_component(thresh_outer)
        mask_inner_inverse = sitk.Not(mask_inner)
        mask_subq = sitk.And(mask_inner_inverse, mask_outer)

        #self.logger.debug("AbFat.PostPredict() - check 3")

        low_values = self.input < -30    
        high_values = self.input > -190

        #self.logger.debug("AbFat.PostPredict() - check 4")

        slab_mask=None
        if not self.region is None:
            out_region_size=[ self.input.GetSize()[0], self.input.GetSize()[1], self.region['Size'][2] ]
            slab_mask = sitku.region_mask(self.input, out_region_size, self.region['Index'])
            #sitk.WriteImage(slab_mask, "slab_mask.nii.gz")

        #self.logger.debug("AbFat.PostPredict() - check 5")

        inner_fat_pre = sitk.And(mask_inner, low_values)
        inner_fat = sitk.And(inner_fat_pre, high_values)
        if not slab_mask is None:
            inner_fat = sitk.Multiply(inner_fat, slab_mask)

        #self.logger.debug("AbFat.PostPredict() - check 6")

        outer_fat_pre = sitk.And(mask_subq, low_values)
        outer_fat = sitk.And(outer_fat_pre, high_values)
        if not slab_mask is None:
            outer_fat = sitk.Multiply(outer_fat, slab_mask)

        self.output=(inner_fat, outer_fat)

def main():

    my_parser = argparse.ArgumentParser(description='Identify contrast enhanced volumes')
    my_parser.add_argument('-m', '--model',  type=str, help='model weights', required=True)
    my_parser.add_argument('-i', '--input',  type=str, help='input nifti volume', required=True)
    my_parser.add_argument('-z', nargs=2, help='first slice,number slices in abdominal slab', required=False)
    my_parser.add_argument('-o', '--output',  type=str, help='output nifti volume', required=True)    
    my_parser.add_argument('-v', '--verbose', dest="verbosity", help="verbose output", action='store_true', default=False)
    args = my_parser.parse_args()

    img = sitk.ReadImage(args.input, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        print("EXITING: Input image is not 3D")
        return(1)
    
    size = list(img.GetSize())
    index = [0] * img.GetDimension()

    AbFats.model =  tf.keras.models.load_model(args.model, custom_objects=get_jabba_custom_objects())

    predictor = AbFats(custom_objects=get_jabba_custom_objects())
    predictor.SetDebugOn()
    predictor.SetImage(img)
    if args.z:
        index[2] = int(args.z[0])
        size[2] = int(args.z[1])
    predictor.SetRegion( {'Index':index, 'Size':size} )
    
    predictor.Update()
    fats = predictor.GetOutput()
    fatSeg = sitk.Add(fats[0], fats[1])
    fatSeg = sitk.Add(fatSeg, fats[1])
    if args.output:
        sitk.WriteImage(fatSeg, args.output)

if __name__=="__main__":
    sys.exit(main())
