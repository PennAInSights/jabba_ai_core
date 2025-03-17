import gradio as gr
import os
from jabba_ai_core.ab_fats import AbFats
from jabba_ai_core.core.jabba_custom_objects import get_jabba_custom_objects
import SimpleITK as sitk
import numpy as np
import logging
import tensorflow as tf

# Setup loggin
logging.basicConfig(level=logging.WARNING)
#logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger("ab_fats_gradio")
logger.setLevel(logging.DEBUG)

# Find model for detecting contrast enhanced CT images
model_name = os.environ['JABBA_FATS_MODEL']
if not os.path.exists(model_name):
    raise ValueError(f"Model {model_name} does not exist")
    exit(1)
else:
    logger.debug(f"Model {model_name} exists")

# Preload the model
AbFats.model =  tf.keras.models.load_model(model_name, custom_objects=get_jabba_custom_objects())
predictor = AbFats()
#predictor.SetDebugOn()
predictor.SetModel(AbFats.model)

# Predict function for Gradio
def ab_fats(filename: str, start: int, n_slices: int, output_filename: str) -> int | list[float] | None:
    logger.debug("ab_fats()")

    if not os.path.isfile(filename):
        logging.error(f"File {filename} does not exist")
        return (None,None)
    img = sitk.ReadImage(filename, sitk.sitkFloat32)
    if img.GetDimension() != 3:
        logging.error("Input image is not 3D")
        return (None,None)     

    size = list(img.GetSize())
    index = [0] * img.GetDimension()
    size[2] = n_slices
    index[2] = start
    predictor.SetImage(img)
    predictor.SetRegion( {'Index':index, 'Size':size} )

    try: 
        predictor.Update()
        seg_imgs = predictor.GetOutput()
        seg_img = sitk.Add(seg_imgs[0], seg_imgs[1])
        seg_img = sitk.Add(seg_img, seg_imgs[1])

        sitk.WriteImage(seg_img, output_filename)
        seg_arr = sitk.GetArrayFromImage(seg_img)
        visceral_vol = np.sum(sitk.GetArrayViewFromImage(seg_imgs[0]))*np.prod(seg_img.GetSpacing())
        sub_vol = np.sum(sitk.GetArrayViewFromImage(seg_imgs[1]))*np.prod(seg_img.GetSpacing())
        return((visceral_vol, sub_vol))
    except:
        return((1,1))


# Setup the Interface
title = "Jabba Abdonimal Fat Segmentation"
desc = "Segment visceral and subcutaneous fat in a CT volume of the abdomen"
art="""# Description
This model segments the visceral and subcutaneous fat in a non-contrast enhanced abdominal CT volume
\n\n# Inputs
* filename = filename of a 3D Nifti volume
* start = first slice in abdominal slab
* n_slices = number of slices in abdominal slab
* output_filename = filename for combined fats (visceral=1, subcutaneious=2) segmentation
\n\n# Outputs
* return_type = (float, float)
  * output0 = volume of visceral fat
  * output1 = volume of subcutaneous fat
# References
* Citation goes here"""

demo = gr.Interface(
    fn=ab_fats,
    inputs=["text", "number", "number", "text"],
    examples=[["/data/ct_abdomen.nii.gz", "78", "98", "/data/ct_abdomen_fats.nii.gz"]],
    outputs=["number", "number"],
    api_name="ab_fats",
    title=title,
    description=desc,
    article=art,
)
demo.launch()
