import SimpleITK as sitk
import jabba_ai_core.core.sitk_utils as sitku
import numpy as np
import logging

# Base class for using images to make deep learning based predictions
class ImagePreprocess():

    def __init__(self, input=None):
        self._input=input
        self._window=1800
        self._level=400
        self._slice_size=[256,256]
        self._pixel_type=sitk.sitkFloat32
        self._debug=False
        self._output=None
        self._orientation='RAI'
        self._logger = logging.getLogger("ImagePreprocess")

    def __str__(self):
        msg=self.__repr__()
        msg=msg+"\n  Image:     "+self.image.__repr__()
        msg=msg+"\n  Window:    "+str(self._window)
        msg=msg+"\n  Level:     "+str(self._level)
        msg=msg+"\n  Window:    "+str(self._slice_size[0])+"x"+str(self.slice_size[1])
        msg=msg+"\n  PixelType: "+sitk.GetPixelIDValueAsString(self._pixel_type)
        return(msg)
    
    @property
    def input(self):
        return self._input
    
    @input.setter
    def input(self, x):
        if not isinstance(x, sitk.Image):
            raise TypeError("Input image must be an sitk.Image")
        if x.GetDimension() != 3:
            raise TypeError("Input image must be 3D")
        self._input=x

    @property
    def window(self):
        return self._window
    
    @window.setter
    def window(self, x):
        if x < 0:
            raise ValueError("Window must be greater than 0")
        self._window=x

    @property
    def level(self):
        return self._level
    
    @level.setter
    def level(self, x):
        self._level=x

    @property
    def slice_size(self):
        return self._slice_size
    
    @slice_size.setter
    def slice_size(self, x):
        if len(x) != 2:
            raise ValueError("Slice size must be a 2 element list")
        self._slice_size=x

    @property
    def pixel_type(self):
        return self._pixel_type
    
    @pixel_type.setter
    def pixel_type(self, x):
        self._pixel_type=x

    @property
    def debug(self):
        return self._debug
    
    @debug.setter
    def debug(self, x):
        self._debug=x

    @property
    def output(self):
        return self._output
    
    @output.setter
    def output(self, x):
        raise ValueError("Output is read only")

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, x):
        self._orientation=x

    def Execute(self):

        self._logger.info("Orientation internally set to "+self._orientation)

        # resize based on how the model was trained
        outSize=[self.slice_size[0],self.slice_size[1],self.input.GetSize()[2]]
        resized = sitku.resize_image(self.input, outSize, interpolation="Linear")

        # window and level
        wmin = self._level - (self._window-1)/2
        wmax = self._level + (self._window-1)/2
        # for unforunate even windows
        if (self._window % 2) == 0:
            wmin = self._level - self._window/2
            wmax = self._level + self._window/2

        scaled = sitk.IntensityWindowing(resized, wmin, wmax, 0.0, 255.0)

        # Reorder voxels to match training data
        oriented = sitku.reorient_image(scaled, self._orientation)

        # Cast to desired output pixel type
        cast = sitk.Cast(oriented, self._pixel_type)

        self._output = cast
        return(self._output)